import functools
import json
import logging
import os
import random
import shutil
import tempfile
from argparse import Namespace

import click
import numpy as np
import torch
import torch.nn.functional as F
from comet_ml import Experiment, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WEIGHTS_NAME
from wikipedia2vec.dump_db import DumpDB

from luke.utils.entity_vocab import MASK_TOKEN, PAD_TOKEN

from ..model_loader import LukeModelLoader
from ..trainer import Trainer
from ..utils import set_seed
from .model import LukeForEntityDisambiguation
from .utils import EntityDisambiguationDataset, convert_documents_to_features

logger = logging.getLogger(__name__)


@click.group(name='entity-disambiguation')
def cli():
    pass


@cli.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.option('-v', '--verbose', is_flag=True)
@click.option('--base-dir', type=click.Path(exists=True))
@click.option('--data-dir', type=click.Path(exists=True), default='data/entity-disambiguation')
@click.option('--titles-file', type=click.Path(exists=True), default='enwiki_20181220_titles.txt')
@click.option('--redirects-file', type=click.Path(exists=True), default='enwiki_20181220_redirects.tsv')
@click.option('--output-dir', default=None, type=click.Path())
@click.option('-t', '--test-set', default=['test_b', 'ace2004', 'aquaint', 'msnbc', 'wikipedia', 'clueweb'],
              multiple=True)
@click.option('--do-train/--no-train', default=False)
@click.option('--do-eval/--no-eval', default=True)
@click.option('--max-candidate-length', default=30)
@click.option('--document-split-mode', default='simple', type=click.Choice(['simple', 'per_mention']))
@click.option('--masked-entity-prob', default=0.7)
@click.option('--use-context-entities/--no-context-entities', default=True)
@click.option('--context-entity-selection-order', default='highest_prob',
              type=click.Choice(['natural', 'random', 'highest_prob']))
@click.option('--train-batch-size', default=1)
@click.option('--gradient-accumulation-steps', default=32)
@click.option('--learning-rate', default=3e-5)
@click.option('--weight-decay', default=0.01)
@click.option('--max-grad-norm', default=1.0)
@click.option('--num-train-epochs', default=2)
@click.option('--warmup-proportion', default=0.2)
@click.option('--grad-avg-on-cpu', is_flag=True)
@click.option('--fix-entity-emb/--update-entity-emb', default=True)
@click.option('--fix-entity-bias/--update-entity-bias', default=True)
def run(**kwargs):
    args = Namespace(**kwargs)
    args.device = torch.device('cuda')
    if not args.base_dir:
        args.base_dir = os.path.dirname(args.model_file)

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_data = LukeModelLoader.load(args.base_dir)
    model_config = model_data.model_config

    dataset = EntityDisambiguationDataset(args.data_dir, args.titles_file, args.redirects_file)
    entity_titles = []
    for data in dataset.get_all_datasets():
        for document in data:
            for mention in document.mentions:
                entity_titles.append(mention.title)
                for candidate in mention.candidates:
                    entity_titles.append(candidate.title)
    entity_titles = frozenset(entity_titles)

    entity_vocab = {PAD_TOKEN: 0, MASK_TOKEN: 1}
    for n, title in enumerate(sorted(entity_titles), 2):
        entity_vocab[title] = n
    model_config.entity_vocab_size = len(entity_vocab)

    logger.info('Model configuration: %s', model_config)

    state_dict = torch.load(args.model_file, map_location='cpu')
    orig_entity_vocab = model_data.entity_vocab
    orig_entity_emb = state_dict['entity_embeddings.entity_embeddings.weight']
    if orig_entity_emb.size(0) != len(entity_vocab):  # detect whether the model is the fine-tuned one
        entity_emb = orig_entity_emb.new_zeros((len(entity_titles) + 2, model_config.hidden_size))
        orig_entity_bias = state_dict['entity_predictions.bias']
        entity_bias = orig_entity_bias.new_zeros(len(entity_titles) + 2)
        for title, index in entity_vocab.items():
            if title in orig_entity_vocab:
                orig_index = orig_entity_vocab[title]
                entity_emb[index] = orig_entity_emb[orig_index]
                entity_bias[index] = orig_entity_bias[orig_index]
        state_dict['entity_embeddings.entity_embeddings.weight'] = entity_emb
        state_dict['entity_embeddings.mask_embedding'] = entity_emb[1].view(1, -1)
        state_dict['entity_predictions.decoder.weight'] = entity_emb
        state_dict['entity_predictions.bias'] = entity_bias
        del orig_entity_bias, entity_emb, entity_bias
    del orig_entity_emb

    model = LukeForEntityDisambiguation(model_config)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)

    def collate_fn(batch, is_eval=False):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            word_ids=create_padded_sequence('word_ids', model_data.tokenizer.pad_token_id),
            word_segment_ids=create_padded_sequence('word_segment_ids', 0),
            word_attention_mask=create_padded_sequence('word_attention_mask', 0),
            entity_ids=create_padded_sequence('entity_ids', 0),
            entity_position_ids=create_padded_sequence('entity_position_ids', -1),
            entity_segment_ids=create_padded_sequence('entity_segment_ids', 0),
            entity_attention_mask=create_padded_sequence('entity_attention_mask', 0),
            entity_candidate_ids=create_padded_sequence('entity_candidate_ids', 0),
        )
        if is_eval:
            ret['document'] = [o.document for o in batch]
            ret['mentions'] = [o.mentions for o in batch]
            ret['target_mention_indices'] = [o.target_mention_indices for o in batch]

        return ret

    if args.do_train:
        logger.info('Training parameters %s', args)

        train_data = convert_documents_to_features(
            dataset.train, model_data.tokenizer, entity_vocab, 'train', 'simple', model_data.max_seq_length,
            args.max_candidate_length, model_data.max_mention_length)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=True)

        logger.info('Fix entity embeddings during training: %s', args.fix_entity_emb)
        if args.fix_entity_emb:
            model.entity_embeddings.entity_embeddings.weight.requires_grad = False
        logger.info('Fix entity bias during training: %s', args.fix_entity_bias)
        if args.fix_entity_bias:
            model.entity_predictions.bias.requires_grad = False

        num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        trainer = EntityDisambiguationTrainer(
            model=model,
            dataloader=train_dataloader,
            device=args.device,
            num_train_steps=num_train_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            warmup_proportion=args.warmup_proportion,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            grad_avg_on_cpu=args.grad_avg_on_cpu,
            masked_entity_prob=args.masked_entity_prob)

        model, global_step, tr_loss = trainer.train()
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

        if args.output_dir:
            logger.info('Saving model to %s', args.output_dir)
            torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    results = {}

    if args.do_eval:
        model.eval()

        for dataset_name in args.test_set:
            logger.info('***** Evaluating: %s *****', dataset_name)
            eval_documents = getattr(dataset, dataset_name)
            eval_data = convert_documents_to_features(
                eval_documents, model_data.tokenizer, entity_vocab, 'eval', args.document_split_mode,
                model_data.max_seq_length, args.max_candidate_length, model_data.max_mention_length)
            eval_dataloader = DataLoader(eval_data, batch_size=1,
                                         collate_fn=functools.partial(collate_fn, is_eval=True))
            predictions_file = None
            if args.output_dir:
                predictions_file = os.path.join(args.output_dir, 'eval_predictions_%s.jsonl' % dataset_name)
            results[dataset_name] = evaluate(args, eval_dataloader, model, entity_vocab, predictions_file)

        if args.output_dir:
            output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
            with open(output_eval_file, 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True)

    return results


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--test-set', default='test_a')
def param_search(config_file, model_file, test_set, **kwargs):
    experiment_cls = functools.partial(Experiment, auto_metric_logging=False, auto_output_logging=None, log_code=False,
                                       log_graph=False)
    opt = Optimizer(config_file, project_name=f"luke-entity-disambiguation-hyperparam-search-5",
                    experiment_class=experiment_cls)

    with open(config_file) as f:
        config = json.load(f)
    base_dir = os.path.dirname(model_file)

    for experiment in opt.get_experiments():
        experiment.disable_mp()
        experiment.log_parameter('model_file', model_file)

        for param_name in config['parameters'].keys():
            kwargs[param_name] = experiment.get_parameter(param_name)

        scores_with_cxt = []
        scores_no_cxt = []
        for seed in range(1, 6):
            output_dir = tempfile.mkdtemp()
            set_seed(seed)

            score = run.callback(model_file=model_file, output_dir=output_dir, base_dir=base_dir, test_set=[test_set],
                                 do_train=True, do_eval=True, use_context_entities=True, **kwargs)[test_set]['f1']
            scores_with_cxt.append(score)
            experiment.log_metric(f'acc_{seed}', score)
            if np.mean(scores_with_cxt) < 0.946:
                shutil.rmtree(output_dir)
                break

            torch.cuda.empty_cache()
            set_seed(seed)

            score = run.callback(model_file=os.path.join(output_dir, WEIGHTS_NAME), output_dir=None, base_dir=base_dir,
                                 test_set=[test_set], do_train=False, do_eval=True, use_context_entities=False,
                                 **kwargs)[test_set]['f1']
            scores_no_cxt.append(score)
            experiment.log_metric(f'acc_no_cxt{seed}', score)
            torch.cuda.empty_cache()

            shutil.rmtree(output_dir)

        experiment.log_metric('acc', np.mean(scores_with_cxt))
        experiment.log_metric('acc_no_cxt', np.mean(scores_no_cxt))
        experiment.end()


param_search.params = param_search.params +\
    [p for p in run.params if p.name not in ('gradient_accumulation_steps', 'masked_entity_prob', 'learning_rate',
                                             'warmup_proportion', 'num_train_epochs', 'do_train', 'do_eval', 'test_set',
                                             'output_dir', 'use_context_entities', 'base_dir')]


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File('w'))
@click.option('--data-dir', type=click.Path(exists=True), default='data/entity-disambiguation')
def create_candidate_list(dump_db_file, out_file, data_dir):
    dump_db = DumpDB(dump_db_file)

    titles = set()
    valid_titles = frozenset(dump_db.titles())

    reader = EntityDisambiguationDataset(data_dir)
    for documents in reader.get_all_datasets():
        for document in documents:
            for mention in document.mentions:
                candidates = mention.candidates
                for candidate in candidates:
                    title = dump_db.resolve_redirect(candidate.title)
                    if title in valid_titles:
                        titles.add(title)

    for title in titles:
        out_file.write(title + '\n')


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File(mode='w'))
def create_title_list(dump_db_file, out_file):
    dump_db = DumpDB(dump_db_file)

    for title in dump_db.titles():
        out_file.write(f'{title}\n')


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File(mode='w'))
def create_redirect_tsv(dump_db_file, out_file):
    dump_db = DumpDB(dump_db_file)

    for src, dest in dump_db.redirects():
        out_file.write(f'{src}\t{dest}\n')


class EntityDisambiguationTrainer(Trainer):
    def __init__(self, masked_entity_prob, *args, **kwargs):
        super(EntityDisambiguationTrainer, self).__init__(*args, **kwargs)
        self._masked_entity_prob = masked_entity_prob

    def _create_model_arguments(self, batch):
        batch['entity_labels'] = batch['entity_ids'].clone()
        for index, entity_length in enumerate(batch['entity_attention_mask'].sum(1).tolist()):
            masked_entity_length = max(1, round(entity_length * self._masked_entity_prob))
            permutated_indices = torch.randperm(entity_length)[:masked_entity_length]
            batch['entity_ids'][index, permutated_indices[:masked_entity_length]] = 1  # [MASK]
            batch['entity_labels'][index, permutated_indices[masked_entity_length:]] = -1

        return batch


def evaluate(args, eval_dataloader, model, entity_vocab, output_file=None):
    predictions = []
    labels = []
    documents = []
    mentions = []
    for item in tqdm(eval_dataloader, leave=False):  # the batch size must be 1
        inputs = {k: v.to(args.device)
                  for k, v in item.items() if k not in ('document', 'mentions', 'target_mention_indices')}
        entity_ids = inputs.pop('entity_ids')
        entity_attention_mask = inputs.pop('entity_attention_mask')
        input_entity_ids = entity_ids.new_full(entity_ids.size(), 1)  # [MASK]
        entity_length = entity_ids.size(1)
        with torch.no_grad():
            if args.use_context_entities:
                result = torch.zeros(entity_length)
                for _ in range(entity_length):
                    logits = model(entity_ids=input_entity_ids, entity_attention_mask=entity_attention_mask,
                                   **inputs)[0]
                    probs = F.softmax(logits, dim=2) * (input_entity_ids == 1).unsqueeze(-1).type_as(logits)
                    max_probs, max_indices = torch.max(probs.squeeze(0), dim=1)
                    if args.context_entity_selection_order == 'highest_prob':
                        target_index = torch.argmax(max_probs, dim=0)
                    elif args.context_entity_selection_order == 'random':
                        target_index = random.choice((input_entity_ids == 1).squeeze(0).nonzero().view(-1).tolist())
                    elif args.context_entity_selection_order == 'natural':
                        target_index = (input_entity_ids == 1).squeeze(0).nonzero().view(-1)[0]
                    input_entity_ids[0, target_index] = max_indices[target_index]
                    result[target_index] = max_indices[target_index]
            else:
                logits = model(entity_ids=input_entity_ids, entity_attention_mask=entity_attention_mask, **inputs)[0]
                result = torch.argmax(logits, dim=2).squeeze(0)

        for index in item['target_mention_indices'][0]:
            predictions.append(result[index].item())
            labels.append(entity_ids[0, index].item())
            documents.append(item['document'][0])
            mentions.append(item['mentions'][0][index])

    num_correct = 0
    num_mentions = 0
    num_mentions_with_candidates = 0
    reverse_entity_vocab = {v: k for k, v in entity_vocab.items()}

    eval_predictions = []
    for prediction, label, document, mention in zip(predictions, labels, documents, mentions):
        if prediction == label:
            num_correct += 1

        assert not (mention.candidates and prediction == 0)
        assert label != 0

        num_mentions += 1
        if mention.candidates:
            num_mentions_with_candidates += 1

            eval_predictions.append(dict(
                document_id=document.id,
                document_length=len(document.words),
                mention_length=len(document.mentions),
                mention=dict(label=mention.title,
                             text=mention.text,
                             span=(mention.start, mention.end),
                             candidate_length=len(mention.candidates)),
                prediction=reverse_entity_vocab[prediction]
            ))

    if output_file:
        with open(output_file, 'w') as f:
            for obj in eval_predictions:
                f.write(json.dumps(obj) + '\n')

    precision = num_correct / num_mentions_with_candidates
    recall = num_correct / num_mentions
    f1 = 2.0 * precision * recall / (precision + recall)

    logger.info('f1: %.3f', f1)
    logger.info('precision: %.3f', precision)
    logger.info('recall: %.3f', recall)
    logger.info('#mentions: %d', num_mentions)
    logger.info('#mentions with candidates: %d', num_mentions_with_candidates)
    logger.info('#correct: %d', num_correct)

    return dict(precision=precision, recall=recall, f1=f1)
