import glob
import logging
import os
import random
import shutil
import string
import tempfile
from argparse import Namespace

import click
import optuna
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME, BertTokenizer
from luke.utils.entity_vocab import MASK_TOKEN, UNK_TOKEN

from ..mention_db import MentionDB
from ..model_loader import LukeModelLoader
from ..trainer import Trainer
from .model import LukeForQuestionAnswering
from .utils import convert_examples_to_features
from .utils_squad import RawResult, read_squad_examples, write_predictions
from .utils_squad_evaluate import EVAL_OPTS
from .utils_squad_evaluate import main as evaluate_on_squad

logger = logging.getLogger(__name__)


@click.group(name='squad')
def cli():
    pass


@cli.command()
@click.argument('mention_db_file', type=click.Path(exists=True))
@click.argument('model_file', type=click.Path(exists=True))
@click.option('--train-file', default='data/squad/train-v2.0.json', type=click.Path(exists=True), required=True)
@click.option('--predict-file', default='data/squad/dev-v2.0.json', type=click.Path(exists=True), required=True)
@click.option('--output-dir', default='squad_' + ''.join(random.choice(string.ascii_letters) for m in range(8)),
              type=click.Path(), required=True)
@click.option('--base-dir', type=click.Path(exists=True, file_okay=False))
@click.option('--version-2-with-negative/--version-1-with-no-negative', default=True)
@click.option('--null-score-diff-threshold', type=float, default=0.0)
@click.option('--max-seq-length', default=512)
@click.option('--max-entity-length', default=128)
@click.option('--max-candidate-length', default=30)
@click.option('--doc-stride', default=128)
@click.option('--max-query-length', default=64)
@click.option('--add-extra-sep-token', is_flag=True)
@click.option('--create-cache', is_flag=True)
@click.option('--do-train/--no-train', default=True)
@click.option('--do-eval/--no-eval', default=True)
@click.option('--train-batch-size', default=1)
@click.option('--eval-batch-size', default=8)
@click.option('--learning-rate', default=15e-6)
@click.option('--gradient-accumulation-steps', default=48)
@click.option('--weight-decay', default=0.01)
@click.option('--lr-layer-decay', default=1.0)
@click.option('--adam-b1', default=0.9)
@click.option('--adam-b2', default=0.98)
@click.option('--adam-eps', default=1e-6)
@click.option('--max-grad-norm', default=0.0)
@click.option('--num-train-epochs', default=2)
@click.option('--warmup-proportion', default=0.06)
@click.option('--n-best-size', default=20)
@click.option('--max-answer-length', default=30)
@click.option('--min-context-entity-prob', default=0.0)
@click.option('--use-softmax-average', is_flag=True)
@click.option('--entity-softmax-temp', default=0.1)
@click.option('--update-params-in-disambi', is_flag=True)
@click.option('--verbose-logging', is_flag=True)
@click.option('--save-steps', default=-1)
@click.option('--eval-all-checkpoints/--no-eval-checkpoints', default=False)
@click.option('--overwrite-cache', is_flag=True)
@click.option('--seed', default=42)
@click.option('--no-cuda', is_flag=True)
@click.option('--local-rank', '--local_rank', default=-1)
@click.option('--fp16', is_flag=True)
@click.option('--fp16-opt-level', default='O2')
@click.option('--fp16-min-loss-scale', default=1)
@click.option('--fp16-max-loss-scale', default=4)
@click.option('--grad-avg-on-cpu', is_flag=True)
@click.option('--fix-entity-emb/--update-entity-emb', default=True)
@click.option('--fix-entity-bias/--update-entity-bias', default=True)
def run(**kwargs):
    args = Namespace(**kwargs)

    if args.no_cuda:
        device = torch.device("cpu")
    elif args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    log_level = logging.INFO
    if args.local_rank not in [-1, 0]:
        log_level = logging.WARN
    logging.getLogger().setLevel(log_level)

    if not args.base_dir:
        args.base_dir = os.path.dirname(args.model_file)

    model_data = LukeModelLoader.load(args.base_dir)
    model_config = model_data.model_config
    logger.info('Model config: %s', model_config)

    tokenizer = model_data.tokenizer
    entity_vocab = model_data.entity_vocab
    mention_db = MentionDB(args.mention_db_file)
    args.max_mention_length = model_data.max_mention_length

    if args.create_cache:
        load_and_cache_examples(args, tokenizer, entity_vocab, mention_db, evaluate=False)
        load_and_cache_examples(args, tokenizer, entity_vocab, mention_db, evaluate=True)
        return

    if args.do_train:
        model = LukeForQuestionAnswering(model_config,
                                         entity_mask_id=entity_vocab[MASK_TOKEN],
                                         entity_unk_id=entity_vocab[UNK_TOKEN],
                                         use_softmax_average=args.use_softmax_average,
                                         entity_softmax_temp=args.entity_softmax_temp,
                                         min_context_entity_prob=args.min_context_entity_prob,
                                         update_params_in_disambi=args.update_params_in_disambi)
        if args.model_file:
            state_dict = torch.load(args.model_file, map_location='cpu')
            # unk_emb = state_dict['entity_embeddings.entity_embeddings.weight'][entity_vocab[UNK_TOKEN]].unsqueeze(0)
            # state_dict['entity_embeddings.unk_embedding'] = unk_emb
            # mask_emb = state_dict['entity_embeddings.entity_embeddings.weight'][entity_vocab[MASK_TOKEN]].unsqueeze(0)
            # state_dict['entity_embeddings.mask_embedding'] = mask_emb
            # state_dict['entity_unk_bias'] = state_dict['entity_predictions.bias'][entity_vocab[UNK_TOKEN]]
            model.load_state_dict(state_dict, strict=False)

        model.to(args.device)
        logger.info("Training parameters %s", args)

        train_dataloader, _, _ = load_and_cache_examples(args, tokenizer, entity_vocab, mention_db, evaluate=False)

        if args.fix_entity_emb:
            model.entity_embeddings.entity_embeddings.weight.requires_grad = False
        if args.fix_entity_bias:
            model.entity_prediction_bias.weight.requires_grad = False

        num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        trainer = Trainer(
            model=model,
            dataloader=train_dataloader,
            device=args.device,
            num_train_steps=num_train_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            adam_b1=args.adam_b1,
            adam_b2=args.adam_b2,
            adam_eps=args.adam_eps,
            warmup_proportion=args.warmup_proportion,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            grad_avg_on_cpu=args.grad_avg_on_cpu,
            local_rank=args.local_rank,
            fp16=args.fp16,
            fp16_opt_level=args.fp16_opt_level,
            fp16_min_loss_scale=args.fp16_min_loss_scale,
            fp16_max_loss_scale=args.fp16_max_loss_scale
        )
        model, global_step, tr_loss = trainer.train()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
        else:
            torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    model = None
    torch.cuda.empty_cache()
    if args.local_rank != -1:
        torch.distributed.barrier()

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME,
                                                                            recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)

        logger.info("Evaluating the following checkpoints: %s", checkpoints)

        for checkpoint_dir in checkpoints:
            global_step = checkpoint_dir.split('-')[-1] if len(checkpoints) > 1 else ""
            model = LukeForQuestionAnswering(model_config,
                                             entity_mask_id=entity_vocab[MASK_TOKEN],
                                             entity_unk_id=entity_vocab[UNK_TOKEN],
                                             use_softmax_average=args.use_softmax_average,
                                             entity_softmax_temp=args.entity_softmax_temp,
                                             min_context_entity_prob=args.min_context_entity_prob,
                                             update_params_in_disambi=args.update_params_in_disambi)
            model.load_state_dict(torch.load(os.path.join(checkpoint_dir, WEIGHTS_NAME), map_location='cpu'))
            model.to(args.device)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            result = evaluate(args, model, tokenizer, entity_vocab, mention_db, prefix=global_step)
            result = {k + '_' + str(global_step) if global_step else k: v for k, v in result.items()}
            results.update(result)

            torch.cuda.empty_cache()

    logger.info("Results: {}".format(results))
    return results


@cli.command()
@click.option('--storage', default='optuna.db', type=click.Path())
@click.option('--study-name', default=None, type=click.Path())
def param_search(model_file, storage, study_name, **kwargs):
    if study_name is None:
        study_name = os.path.basename(model_file) + ''.join(random.choice(string.ascii_letters) for m in range(8))

    logger.info('study name: %s', study_name)

    def objective(trial):
        trial.set_user_attr('model_file', model_file)

        output_dir = tempfile.mkdtemp()
        lr = trial.suggest_uniform('lr', 1e-5, 1e-1)
        score = run.callback(model_file=model_file, output_dir=output_dir, base_dir=base_dir, test_set=[test_set],
                                do_train=True, do_eval=True, use_context_entities=True, **kwargs)[test_set]['f1']

        torch.cuda.empty_cache()
        shutil.rmtree(output_dir)

    study = optuna.create_study(direction='maximize', load_if_exists=True)
    study.optimize(objective)


# param_search.params = param_search.params +\
#     [p for p in run.params if p.name not in ('do_train', 'do_eval', 'test_set', 'gradient_accumulation_steps', 'masked_entity_prob', 'learning_rate',
#                                              'warmup_proportion', 'num_train_epochs',
#                                              'output_dir', 'use_context_entities', 'base_dir')]


def evaluate(args, model, tokenizer, entity_vocab, mention_db, prefix=""):
    dataloader, examples, features = load_and_cache_examples(args, tokenizer, entity_vocab, mention_db, evaluate=True)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    all_results = []
    for batch in tqdm(dataloader, desc="Eval"):
        model.eval()
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != 'example_indices'}
        with torch.no_grad():
            outputs = model(**inputs)

        for i, example_index in enumerate(batch['example_indices']):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id, start_logits=outputs[0][i].detach().cpu().tolist(),
                                         end_logits=outputs[1][i].detach().cpu().tolist()))

    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    do_lower_case = False
    if isinstance(tokenizer, BertTokenizer):
        do_lower_case = tokenizer.basic_tokenizer.do_lower_case

    write_predictions(examples, features, all_results, args.n_best_size, args.max_answer_length, do_lower_case,
                      output_prediction_file, output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                      args.version_2_with_negative, args.null_score_diff_threshold, tokenizer)

    return evaluate_on_squad(EVAL_OPTS(data_file=args.predict_file,
                                       pred_file=output_prediction_file,
                                       na_prob_file=output_null_log_odds_file))


def load_and_cache_examples(args, tokenizer, entity_vocab, mention_db, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    input_file = args.predict_file if evaluate else args.train_file
    examples = read_squad_examples(input_file=input_file, is_training=not evaluate,
                                   version_2_with_negative=args.version_2_with_negative)
    cache_file = os.path.join(os.path.dirname(input_file), 'cached_' + '_'.join((
        os.path.basename(input_file),
        tokenizer.__class__.__name__,
        str(len(entity_vocab)),
        os.path.basename(mention_db.mention_db_file),
        str(args.max_seq_length),
        str(args.max_mention_length),
        str(args.max_candidate_length),
        str(args.doc_stride),
        str(args.max_query_length),
        str(args.add_extra_sep_token),
    )))
    if os.path.exists(cache_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cache_file)
        features = torch.load(cache_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        features = convert_examples_to_features(
            examples, tokenizer, entity_vocab, mention_db, args.max_seq_length, args.max_mention_length,
            args.max_candidate_length, args.doc_stride, args.max_query_length, args.add_extra_sep_token, not evaluate)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cache_file)
            torch.save(features, cache_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o[1], attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            word_ids=create_padded_sequence('word_ids', tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence('word_attention_mask', 0),
            word_segment_ids=create_padded_sequence('word_segment_ids', 0),
            entity_candidate_ids=create_padded_sequence('entity_candidate_ids', 0)[:, :args.max_entity_length, :],
            entity_attention_mask=create_padded_sequence('entity_attention_mask', 0)[:, :args.max_entity_length],
            entity_position_ids=create_padded_sequence('entity_position_ids', -1)[:, :args.max_entity_length, :],
            entity_segment_ids=create_padded_sequence('entity_segment_ids', 0)[:, :args.max_entity_length],
        )

        if evaluate:
            ret['example_indices'] = torch.tensor([o[0] for o in batch], dtype=torch.long)
        else:
            ret['start_positions'] = torch.tensor([o[1].start_position for o in batch], dtype=torch.long)
            ret['end_positions'] = torch.tensor([o[1].end_position for o in batch], dtype=torch.long)

        return ret

    if evaluate:
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn)
    else:
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size,
                                collate_fn=collate_fn)

    return dataloader, examples, features
