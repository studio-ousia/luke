import json
import logging
import os
from argparse import Namespace
from functools import partial

import click
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME

from luke.utils.entity_vocab import MASK_TOKEN

from ..trainer import Trainer, trainer_args
from ..utils import set_seed
from ..word_entity_model import word_entity_model_args
from .model import LukeForRelationClassification
from .utils import ENTITY_TYPES, convert_examples_to_features, DatasetProcessor

logger = logging.getLogger(__name__)


@click.group(name='relation-classification')
def cli():
    pass


@cli.command()
@click.option('--data-dir', default='data/tacred', type=click.Path(exists=True))
@click.option('--do-train/--no-train', default=True)
@click.option('--train-batch-size', default=1)
@click.option('--num-train-epochs', default=2.0)
@click.option('--do-eval/--no-eval', default=True)
@click.option('--eval-batch-size', default=8)
@click.option('--eval-set', default=['dev'], type=click.Choice(['dev', 'test']), multiple=True)
@click.option('--evaluate-every-epoch', is_flag=True)
@click.option('--use-entity-type-token', is_flag=True)
@click.option('--use-hidden-layer', is_flag=True)
@click.option('--use-difference-feature', is_flag=True)
@click.option('--fix-entity-emb', is_flag=True)
@click.option('--dropout-prob', default=0.1)
@click.option('--seed', default=42)
@word_entity_model_args
@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    task_args.update(common_args)
    args = Namespace(**task_args)

    set_seed(args.seed)

    args.experiment.log_parameters({p.name: getattr(args, p.name) for p in run.params})

    if 'roberta' in args.bert_model_name:
        args.tokenizer.tokenize = partial(args.tokenizer.tokenize, add_prefix_space=True)

    entity_emb = args.model_weights['entity_embeddings.entity_embeddings.weight']
    mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)
    if args.use_entity_type_token:
        args.model_config.entity_vocab_size = len(ENTITY_TYPES) + 1
        mask_emb = mask_emb.expand(len(ENTITY_TYPES), -1)
        args.model_weights['entity_embeddings.entity_embeddings.weight'] = torch.cat([entity_emb[:1], mask_emb])
    else:
        args.model_config.entity_vocab_size = 2
        args.model_weights['entity_embeddings.entity_embeddings.weight'] = torch.cat([entity_emb[:1], mask_emb])

    train_dataloader, _, _, label_list = load_and_cache_examples(args, fold='train')
    num_labels = len(label_list)

    if args.do_train:
        model = LukeForRelationClassification(args, num_labels)
        model.load_state_dict(args.model_weights, strict=False)
        model.to(args.device)

        if args.fix_entity_emb:
            model.entity_embeddings.entity_embeddings.weight.requires_grad = False

        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

        if args.evaluate_every_epoch:
            def step_callback(model, global_step, tqdm_ins):
                if global_step % num_train_steps_per_epoch == 0 and args.local_rank in (0, -1):
                    epoch = int(global_step / num_train_steps_per_epoch - 1)
                    for eval_set in args.eval_set:
                        results = evaluate(args, model, fold=eval_set)
                        args.experiment.log_metrics(
                            {f'{eval_set}_{k}_epoch{epoch}': v for k, v in results.items()}, epoch=epoch)
                        tqdm_ins.write(f'{eval_set}: {str(results)}')
                    model.train()
        else:
            step_callback = None

        trainer = Trainer(args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps,
                          step_callback=step_callback)
        trainer.train()

    if args.do_train and args.local_rank in (0, -1):
        logger.info('Saving model checkpoint to %s', args.output_dir)
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
        else:
            torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))

    if args.local_rank not in (0, -1):
        return {}

    model = None
    torch.cuda.empty_cache()

    results = {}
    if args.do_eval:
        model = LukeForRelationClassification(args, num_labels)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location='cpu'))
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(args.device)

        for eval_set in args.eval_set:
            results.update({f'{eval_set}_{k}':v for k, v in evaluate(args, model, fold=eval_set).items()})

    print(results)
    args.experiment.log_metrics(results)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f)

    return results


def evaluate(args, model, fold='dev'):
    dataloader, _, _, _ = load_and_cache_examples(args, fold=fold)
    predictions = []
    labels = []

    model.eval()
    for batch in tqdm(dataloader, desc=fold):
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != 'label'}
        with torch.no_grad():
            logits = model(**inputs)

        predictions.extend(logits.detach().cpu().numpy().argmax(axis=1))
        labels.extend(batch['label'].to('cpu').tolist())

    num_predicted_labels = 0
    num_gold_labels = 0
    num_correct_labels = 0

    for label, prediction in zip(labels, predictions):
        if prediction != 0:
            num_predicted_labels += 1
        if label != 0:
            num_gold_labels += 1
            if prediction == label:
                num_correct_labels += 1

    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.
    recall = num_correct_labels / num_gold_labels
    if recall == 0.:
        f1 = 0.
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return dict(precision=precision, recall=recall, f1=f1)


def load_and_cache_examples(args, fold='train'):
    if args.local_rank not in (-1, 0) and fold == 'train':
        torch.distributed.barrier()

    processor = DatasetProcessor()
    if fold == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif fold == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    label_list = processor.get_label_list(args.data_dir)

    bert_model_name = args.model_config.bert_model_name

    cache_file = os.path.join(args.data_dir, 'cached_' + '_'.join((
        bert_model_name.split('-')[0],
        str(len(args.entity_vocab)),
        str(args.max_mention_length),
        str(args.use_entity_type_token),
        fold
    )) + '.pkl')
    if os.path.exists(cache_file):
        logger.info("Loading features from cached file %s", cache_file)
        features = torch.load(cache_file)
    else:
        logger.info("Creating features from dataset file")

        features = convert_examples_to_features(
            examples, label_list, args.tokenizer, args.max_mention_length, args.use_entity_type_token)

        if args.local_rank in (-1, 0):
            torch.save(features, cache_file)

    if args.local_rank == 0 and fold == 'train':
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        return dict(
            word_ids=create_padded_sequence('word_ids', args.tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence('word_attention_mask', 0),
            word_segment_ids=create_padded_sequence('word_segment_ids', 0),
            entity_ids=create_padded_sequence('entity_ids', 0),
            entity_attention_mask=create_padded_sequence('entity_attention_mask', 0),
            entity_position_ids=create_padded_sequence('entity_position_ids', -1),
            entity_segment_ids=create_padded_sequence('entity_segment_ids', 0),
            label=torch.tensor([o.label for o in batch], dtype=torch.long)
        )

    if fold in ('dev', 'test'):
        dataloader = DataLoader(features, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(features, sampler=sampler, batch_size=args.train_batch_size,
                                collate_fn=collate_fn)

    return dataloader, examples, features, label_list
