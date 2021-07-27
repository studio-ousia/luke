import json
import logging
import multiprocessing
import os
from argparse import Namespace

import click
import joblib
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME, BertTokenizer
from wikipedia2vec.dump_db import DumpDB

from ..utils import set_seed
from ..utils.mention_db import MentionDB
from ..utils.trainer import Trainer, trainer_args
from .model import LukeForReadingComprehension
from .utils.dataset import SquadV1Processor, SquadV2Processor
from .utils.feature import convert_examples_to_features
from .utils.result_writer import Result, write_predictions
from .utils.squad_eval import EVAL_OPTS as SQUAD_EVAL_OPTS
from .utils.squad_eval import main as evaluate_on_squad
from .utils.wiki_link_db import WikiLinkDB

logger = logging.getLogger(__name__)


@click.group(name="reading-comprehension")
def cli():
    pass


@cli.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("mention_db_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
@click.pass_obj
def build_wiki_link_db(common_args, dump_db_file, mention_db_file, **kwargs):
    dump_db = DumpDB(dump_db_file)
    mention_db = MentionDB(mention_db_file)
    WikiLinkDB.build(dump_db, mention_db, **kwargs)


@cli.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.option("--compress", default=3)
def generate_redirect_file(dump_db_file, out_file, compress):
    data = {k: v for k, v in DumpDB(dump_db_file).redirects()}
    joblib.dump(data, out_file, compress=compress)


@cli.command()
@click.option("--checkpoint-file", type=click.Path(exists=True))
@click.option("--data-dir", default="data/squad", type=click.Path(exists=True))
@click.option("--do-eval/--no-eval", default=True)
@click.option("--do-train/--no-train", default=True)
@click.option("--doc-stride", default=128)
@click.option("--eval-batch-size", default=32)
@click.option("--link-redirects-file", type=click.Path(exists=True), default="enwiki_20160305_redirects.pkl")
@click.option("--max-answer-length", default=30)
@click.option("--max-entity-length", default=128)
@click.option("--max-query-length", default=64)
@click.option("--max-seq-length", default=512)
@click.option("--min-mention-link-prob", default=0.01)
@click.option("--model-redirects-file", type=click.Path(exists=True), default="enwiki_20181220_redirects.pkl")
@click.option("--n-best-size", default=20)
@click.option("--no-entity", is_flag=True, default=False)
@click.option("--null-score-diff-threshold", type=float, default=0.0)
@click.option("--num-train-epochs", default=2)
@click.option("--seed", default=14)
@click.option("--train-batch-size", default=2)
@click.option("--wiki-link-db-file", type=click.Path(exists=True), default="enwiki_20160305.pkl")
@click.option("--with-negative/--no-negative", default=True)
@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    task_args.update(common_args)
    args = Namespace(**task_args)

    set_seed(args.seed)

    args.experiment.log_parameters({p.name: getattr(args, p.name) for p in run.params})

    args.wiki_link_db = WikiLinkDB(args.wiki_link_db_file)
    args.model_redirect_mappings = joblib.load(args.model_redirects_file)
    args.link_redirect_mappings = joblib.load(args.link_redirects_file)

    if args.do_train:
        model = LukeForReadingComprehension(args)
        model.load_state_dict(args.model_weights, strict=False)
        model.to(args.device)

        model.entity_embeddings.entity_embeddings.weight.requires_grad = False

        train_dataloader, _, _, _ = load_examples(args, evaluate=False)

        num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        trainer = Trainer(args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps)
        trainer.train()

    if args.do_train and args.local_rank in (0, -1):
        if hasattr(model, "module"):
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
        else:
            torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))

    if args.local_rank not in (0, -1):
        return {}

    model = None
    torch.cuda.empty_cache()

    results = {}

    if args.do_eval:
        model = LukeForReadingComprehension(args)
        if args.checkpoint_file:
            model.load_state_dict(torch.load(args.checkpoint_file, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
        model.to(args.device)

        result = evaluate(args, model, prefix="")
        results.update(result)

    logger.info("Results: %s", json.dumps(results, indent=2, sort_keys=True))
    args.experiment.log_metrics(results)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)

    return results


def evaluate(args, model, prefix=""):
    dataloader, examples, features, processor = load_examples(args, evaluate=True)
    all_results = []
    for batch in tqdm(dataloader, desc="eval"):
        model.eval()
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "example_indices"}
        with torch.no_grad():
            outputs = model(**inputs)

        for i, example_index in enumerate(batch["example_indices"]):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            start_logits, end_logits = [o[i].detach().cpu().tolist() for o in outputs]
            all_results.append(Result(unique_id, start_logits, end_logits))

    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    do_lower_case = False
    if isinstance(args.tokenizer, BertTokenizer):
        do_lower_case = args.tokenizer.basic_tokenizer.do_lower_case

    write_predictions(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,
        args.with_negative,
        args.null_score_diff_threshold,
        args.tokenizer,
    )

    return evaluate_on_squad(
        SQUAD_EVAL_OPTS(
            os.path.join(args.data_dir, processor.dev_file),
            pred_file=output_prediction_file,
            na_prob_file=output_null_log_odds_file,
        )
    )


def load_examples(args, evaluate=False):
    if args.local_rank not in (-1, 0) and not evaluate:
        torch.distributed.barrier()

    if args.with_negative:
        processor = SquadV2Processor()
    else:
        processor = SquadV1Processor()

    if evaluate:
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    bert_model_name = args.model_config.bert_model_name

    segment_b_id = 1
    add_extra_sep_token = False
    if "roberta" in bert_model_name:
        segment_b_id = 0
        add_extra_sep_token = True

    logger.info("Creating features from the dataset...")
    features = convert_examples_to_features(
        examples=examples,
        tokenizer=args.tokenizer,
        entity_vocab=args.entity_vocab,
        wiki_link_db=args.wiki_link_db,
        model_redirect_mappings=args.model_redirect_mappings,
        link_redirect_mappings=args.link_redirect_mappings,
        max_seq_length=args.max_seq_length,
        max_mention_length=args.max_mention_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        min_mention_link_prob=args.min_mention_link_prob,
        segment_b_id=segment_b_id,
        add_extra_sep_token=add_extra_sep_token,
        is_training=not evaluate,
    )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o[1], attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_ids=create_padded_sequence("entity_ids", 0)[:, : args.max_entity_length],
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0)[:, : args.max_entity_length],
            entity_position_ids=create_padded_sequence("entity_position_ids", -1)[:, : args.max_entity_length, :],
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0)[:, : args.max_entity_length],
        )
        if args.no_entity:
            ret["entity_attention_mask"].fill_(0)

        if evaluate:
            ret["example_indices"] = torch.tensor([o[0] for o in batch], dtype=torch.long)
        else:
            ret["start_positions"] = torch.tensor([o[1].start_positions[0] for o in batch], dtype=torch.long)
            ret["end_positions"] = torch.tensor([o[1].end_positions[0] for o in batch], dtype=torch.long)

        return ret

    if evaluate:
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn)
    else:
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(
            list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
        )

    return dataloader, examples, features, processor
