import json
import logging
import os
from argparse import Namespace
from collections import defaultdict

import click
import seqeval.metrics
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME

from luke.utils.entity_vocab import MASK_TOKEN

from ..utils import set_seed
from ..utils.trainer import Trainer, trainer_args
from .model import LukeForNamedEntityRecognition
from .utils import CoNLLProcessor, convert_examples_to_features

logger = logging.getLogger(__name__)


@click.group(name="ner")
def cli():
    pass


@cli.command()
@click.option("--checkpoint-file", type=click.Path(exists=True))
@click.option("--data-dir", default="data/conll_2003", type=click.Path(exists=True))
@click.option("--do-train/--no-train", default=True)
@click.option("--do-eval/--no-eval", default=True)
@click.option("--eval-batch-size", default=32)
@click.option("--max-entity-length", default=128)
@click.option("--max-mention-length", default=16)
@click.option("--max-seq-length", default=512)
@click.option("--no-entity-feature", is_flag=True)
@click.option("--no-word-feature", is_flag=True)
@click.option("--train-batch-size", default=2)
@click.option("--num-train-epochs", default=5.0)
@click.option("--seed", default=35)
@click.option("--train-on-dev-set", is_flag=True)
@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    common_args.update(task_args)
    args = Namespace(**common_args)

    set_seed(args.seed)

    args.experiment.log_parameters({p.name: getattr(args, p.name) for p in run.params})

    args.model_config.entity_vocab_size = 2
    entity_emb = args.model_weights["entity_embeddings.entity_embeddings.weight"]
    mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)
    args.model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])

    train_dataloader, _, _, processor = load_examples(args, "train")
    results = {}

    if args.do_train:
        model = LukeForNamedEntityRecognition(args, len(processor.get_labels()))
        model.load_state_dict(args.model_weights, strict=False)
        model.to(args.device)

        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

        trainer = Trainer(args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps)
        trainer.train()

    if args.do_train and args.local_rank in (0, -1):
        logger.info("Saving the model checkpoint to %s", args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))

    if args.local_rank not in (0, -1):
        return {}

    model = None
    torch.cuda.empty_cache()

    if args.do_eval:
        model = LukeForNamedEntityRecognition(args, len(processor.get_labels()))
        if args.checkpoint_file:
            model.load_state_dict(torch.load(args.checkpoint_file, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
        model.to(args.device)

        dev_output_file = os.path.join(args.output_dir, "dev_predictions.txt")
        test_output_file = os.path.join(args.output_dir, "test_predictions.txt")
        results.update({f"dev_{k}": v for k, v in evaluate(args, model, "dev", dev_output_file).items()})
        results.update({f"test_{k}": v for k, v in evaluate(args, model, "test", test_output_file).items()})

    logger.info("Results: %s", json.dumps(results, indent=2, sort_keys=True))
    args.experiment.log_metrics(results)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)

    return results


def evaluate(args, model, fold, output_file=None):
    dataloader, examples, features, processor = load_examples(args, fold)
    label_list = processor.get_labels()
    all_predictions = defaultdict(dict)

    for batch in tqdm(dataloader, desc="Eval"):
        model.eval()
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
        with torch.no_grad():
            logits = model(**inputs)

        for i, feature_index in enumerate(batch["feature_indices"]):
            feature = features[feature_index.item()]
            for j, span in enumerate(feature.original_entity_spans):
                if span is not None:
                    all_predictions[feature.example_index][span] = logits[i, j].detach().cpu().max(dim=0)

    assert len(all_predictions) == len(examples)

    sent_words_list = []
    sent_labels_list = []
    sent_predictions_list = []

    for example_index, example in enumerate(examples):
        predictions = all_predictions[example_index]
        doc_results = []
        for span, (max_logit, max_index) in predictions.items():
            if max_index != 0:
                doc_results.append((max_logit.item(), span, label_list[max_index.item()]))

        predicted_sequence = ["O"] * len(example.words)
        for _, span, label in sorted(doc_results, key=lambda o: o[0], reverse=True):
            if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
                predicted_sequence[span[0]] = "B-" + label
                if span[1] - span[0] > 1:
                    predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

        for sent_index in range(len(example.sentence_boundaries) - 1):
            sent_start, sent_end = example.sentence_boundaries[sent_index : sent_index + 2]
            sent_words_list.append(example.words[sent_start:sent_end])
            sent_predictions_list.append(predicted_sequence[sent_start:sent_end])
            sent_labels_list.append(example.labels[sent_start:sent_end])

    # convert IOB2 -> IOB1
    prev_type = None
    for sent_predictions in sent_predictions_list:
        for n, label in enumerate(sent_predictions):
            if label[0] == "B" and label[2:] != prev_type:
                sent_predictions[n] = "I" + label[1:]
            prev_type = label[2:]

    if output_file:
        with open(output_file, "w") as f:
            for (sent_words, sent_predictions, sent_labels) in zip(
                sent_words_list, sent_predictions_list, sent_labels_list
            ):
                for word, prediction, label in zip(sent_words, sent_predictions, sent_labels):
                    f.write(f"{word} {label} {prediction}\n")
                f.write("\n")

    print(seqeval.metrics.classification_report(sent_labels_list, sent_predictions_list, digits=4))

    return dict(
        f1=seqeval.metrics.f1_score(sent_labels_list, sent_predictions_list),
        precision=seqeval.metrics.precision_score(sent_labels_list, sent_predictions_list),
        recall=seqeval.metrics.recall_score(sent_labels_list, sent_predictions_list),
    )


def load_examples(args, fold):
    if args.local_rank not in (-1, 0) and fold == "train":
        torch.distributed.barrier()

    processor = CoNLLProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    if fold == "train" and args.train_on_dev_set:
        examples += processor.get_dev_examples(args.data_dir)

    label_list = processor.get_labels()

    logger.info("Creating features from the dataset...")
    features = convert_examples_to_features(
        examples, label_list, args.tokenizer, args.max_seq_length, args.max_entity_length, args.max_mention_length
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

        ret = dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_start_positions=create_padded_sequence("entity_start_positions", 0),
            entity_end_positions=create_padded_sequence("entity_end_positions", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
        )
        if args.no_entity_feature:
            ret["entity_ids"].fill_(0)
            ret["entity_attention_mask"].fill_(0)

        if fold == "train":
            ret["labels"] = create_padded_sequence("labels", -1)
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
