import json
import logging
import os
from argparse import Namespace
from collections import defaultdict

import click
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME

from luke.utils.entity_vocab import MASK_TOKEN

from ..utils import set_seed
from ..utils.trainer import Trainer, trainer_args
from .model import LukeForEntitySpanQA
from .record_eval import evaluate as evaluate_on_record
from .utils import (
    HIGHLIGHT_TOKEN,
    PLACEHOLDER_TOKEN,
    ENTITY_MARKER_TOKEN,
    RecordProcessor,
    convert_examples_to_features,
)

logger = logging.getLogger(__name__)


@click.group(name="entity-span-qa")
def cli():
    pass


@cli.command()
@click.option("--checkpoint-file", type=click.Path(exists=True))
@click.option("--data-dir", default="data/record", type=click.Path(exists=True))
@click.option("--doc-stride", default=128)
@click.option("--do-eval/--no-eval", default=True)
@click.option("--do-train/--no-train", default=True)
@click.option("--eval-batch-size", default=32)
@click.option("--max-query-length", default=90)
@click.option("--max-seq-length", default=512)
@click.option("--num-train-epochs", default=2.0)
@click.option("--seed", default=4)
@click.option("--train-batch-size", default=1)
@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    task_args.update(common_args)
    args = Namespace(**task_args)

    set_seed(args.seed)

    args.experiment.log_parameters({p.name: getattr(args, p.name) for p in run.params})

    args.model_config.vocab_size += 3
    word_emb = args.model_weights["embeddings.word_embeddings.weight"]
    highlight_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
    placeholder_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
    marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["*"])[0]].unsqueeze(0)
    args.model_weights["embeddings.word_embeddings.weight"] = torch.cat(
        [word_emb, highlight_emb, placeholder_emb, marker_emb]
    )
    args.tokenizer.add_special_tokens(
        dict(additional_special_tokens=[HIGHLIGHT_TOKEN, PLACEHOLDER_TOKEN, ENTITY_MARKER_TOKEN])
    )

    args.model_config.entity_vocab_size = 2
    entity_emb = args.model_weights["entity_embeddings.entity_embeddings.weight"]
    mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)
    args.model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])

    results = {}
    if args.do_train:
        model = LukeForEntitySpanQA(args)
        model.load_state_dict(args.model_weights, strict=False)
        model.to(args.device)

        train_dataloader, _, _, _ = load_examples(args, "train")

        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

        best_dev_score = [-1]
        best_weights = [None]

        def step_callback(model, global_step):
            if global_step % num_train_steps_per_epoch == 0 and args.local_rank in (0, -1):
                epoch = int(global_step / num_train_steps_per_epoch - 1)
                dev_results = evaluate(args, model, fold="dev")
                args.experiment.log_metrics({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()}, epoch=epoch)
                results.update({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()})
                tqdm.write("dev: " + str(dev_results))

                if dev_results["exact_match"] > best_dev_score[0]:
                    if hasattr(model, "module"):
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in model.module.state_dict().items()}
                    else:
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                    best_dev_score[0] = dev_results["exact_match"]
                    results["best_epoch"] = epoch

                model.train()

        trainer = Trainer(
            args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps, step_callback=step_callback
        )
        trainer.train()

    if args.do_train and args.local_rank in (0, -1):
        logger.info("Saving the model checkpoint to %s", args.output_dir)
        torch.save(best_weights[0], os.path.join(args.output_dir, WEIGHTS_NAME))

    if args.local_rank not in (0, -1):
        return {}

    model = None
    torch.cuda.empty_cache()

    if args.do_eval:
        model = LukeForEntitySpanQA(args)
        if args.checkpoint_file:
            model.load_state_dict(torch.load(args.checkpoint_file, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
        model.to(args.device)

        output_file = os.path.join(args.output_dir, "predictions.json")
        results.update({f"dev_{k}": v for k, v in evaluate(args, model, fold="dev", output_file=output_file).items()})

    logger.info("Results: %s", json.dumps(results, indent=2, sort_keys=True))
    args.experiment.log_metrics(results)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)

    return results


def evaluate(args, model, fold="dev", output_file=None):
    dataloader, examples, features, processor = load_examples(args, fold)
    doc_predictions = defaultdict(list)
    for batch in tqdm(dataloader, desc="Eval"):
        model.eval()
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
        with torch.no_grad():
            logits = model(**inputs)

        for i, feature_index in enumerate(batch["feature_indices"]):
            feature = features[feature_index.item()]
            max_logit, max_index = logits[i].detach().max(dim=0)
            example_id = examples[feature.example_index].qas_id
            entity = feature.entities[max_index.item()]
            doc_predictions[example_id].append((max_logit, entity))

    predictions = {k: sorted(v, key=lambda o: o[0])[-1][1]["text"] for k, v in doc_predictions.items()}
    if output_file:
        with open(output_file, "w") as f:
            json.dump(predictions, f)

    with open(os.path.join(args.data_dir, processor.dev_file)) as f:
        dev_data = json.load(f)["data"]

    return evaluate_on_record(dev_data, predictions)[0]


def load_examples(args, fold):
    if args.local_rank not in (-1, 0) and fold == "train":
        torch.distributed.barrier()

    processor = RecordProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    else:
        examples = processor.get_dev_examples(args.data_dir)

    bert_model_name = args.model_config.bert_model_name

    if "roberta" in bert_model_name:
        segment_b_id = 0
        add_extra_sep_token = True
    else:
        segment_b_id = 1
        add_extra_sep_token = False

    logger.info("Creating features from the dataset...")
    features = convert_examples_to_features(
        examples,
        args.tokenizer,
        args.max_seq_length,
        args.max_mention_length,
        args.doc_stride,
        args.max_query_length,
        segment_b_id,
        add_extra_sep_token,
    )

    if args.local_rank == 0 and fold == "train":
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value):
            if isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.long) for o in batch]
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in target]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        entity_ids = []
        entity_segment_ids = []
        entity_attention_mask = []
        entity_position_ids = []

        for _, item in batch:
            entity_length = len(item.entity_position_ids) + 1
            entity_ids.append([1] * entity_length)
            entity_segment_ids.append([0] + [segment_b_id] * (entity_length - 1))
            entity_attention_mask.append([1] * entity_length)
            entity_position_ids.append(item.placeholder_position_ids + item.entity_position_ids)
            if entity_length == 1:
                entity_ids[-1].append(0)
                entity_segment_ids[-1].append(0)
                entity_attention_mask[-1].append(0)
                entity_position_ids[-1].append([-1] * args.max_mention_length)

        ret = dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_ids=create_padded_sequence(entity_ids, 0),
            entity_segment_ids=create_padded_sequence(entity_segment_ids, 0),
            entity_attention_mask=create_padded_sequence(entity_attention_mask, 0),
            entity_position_ids=create_padded_sequence(entity_position_ids, -1),
        )
        if fold == "train":
            ret["labels"] = create_padded_sequence("labels", 0)
        else:
            ret["feature_indices"] = torch.tensor([o[0] for o in batch], dtype=torch.long)

        return ret

    if fold == "train":
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(
            list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, processor
