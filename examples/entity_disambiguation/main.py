import functools
import json
import logging
import os
import random
from argparse import Namespace

import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WEIGHTS_NAME
from wikipedia2vec.dump_db import DumpDB

from luke.utils.entity_vocab import MASK_TOKEN, PAD_TOKEN

from ..utils.trainer import Trainer, trainer_args
from ..utils import set_seed
from .model import LukeForEntityDisambiguation
from .utils import EntityDisambiguationDataset, convert_documents_to_features

logger = logging.getLogger(__name__)


@click.group(name="entity-disambiguation")
def cli():
    pass


@cli.command()
@click.option("--data-dir", type=click.Path(exists=True), default="data/entity_disambiguation")
@click.option(
    "-t",
    "--test-set",
    default=["test_b", "test_b_ppr", "ace2004", "aquaint", "msnbc", "wikipedia", "clueweb"],
    multiple=True,
)
@click.option("--do-train/--no-train", default=False)
@click.option("--do-eval/--no-eval", default=True)
@click.option("--num-train-epochs", default=2)
@click.option("--train-batch-size", default=1)
@click.option("--max-seq-length", default=512)
@click.option("--max-candidate-length", default=30)
@click.option("--masked-entity-prob", default=0.9)
@click.option("--use-context-entities/--no-context-entities", default=True)
@click.option(
    "--context-entity-selection-order", default="highest_prob", type=click.Choice(["natural", "random", "highest_prob"])
)
@click.option("--document-split-mode", default="per_mention", type=click.Choice(["simple", "per_mention"]))
@click.option("--fix-entity-emb/--update-entity-emb", default=True)
@click.option("--fix-entity-bias/--update-entity-bias", default=True)
@click.option("--seed", default=1)
@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    task_args.update(common_args)
    args = Namespace(**task_args)

    set_seed(args.seed)

    dataset = EntityDisambiguationDataset(args.data_dir)
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

    model_config = args.model_config
    model_config.entity_vocab_size = len(entity_vocab)

    model_weights = args.model_weights
    orig_entity_vocab = args.entity_vocab
    orig_entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]
    if orig_entity_emb.size(0) != len(entity_vocab):  # detect whether the model is fine-tuned
        entity_emb = orig_entity_emb.new_zeros((len(entity_titles) + 2, model_config.hidden_size))
        orig_entity_bias = model_weights["entity_predictions.bias"]
        entity_bias = orig_entity_bias.new_zeros(len(entity_titles) + 2)
        for title, index in entity_vocab.items():
            if title in orig_entity_vocab:
                orig_index = orig_entity_vocab[title]
                entity_emb[index] = orig_entity_emb[orig_index]
                entity_bias[index] = orig_entity_bias[orig_index]
        model_weights["entity_embeddings.entity_embeddings.weight"] = entity_emb
        model_weights["entity_embeddings.mask_embedding"] = entity_emb[1].view(1, -1)
        model_weights["entity_predictions.decoder.weight"] = entity_emb
        model_weights["entity_predictions.bias"] = entity_bias
        del orig_entity_bias, entity_emb, entity_bias
    del orig_entity_emb

    model = LukeForEntityDisambiguation(model_config)
    model.load_state_dict(model_weights, strict=False)
    model.to(args.device)

    def collate_fn(batch, is_eval=False):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
        )
        ret["entity_candidate_ids"] = create_padded_sequence("entity_candidate_ids", 0)

        if is_eval:
            ret["document"] = [o.document for o in batch]
            ret["mentions"] = [o.mentions for o in batch]
            ret["target_mention_indices"] = [o.target_mention_indices for o in batch]

        return ret

    if args.do_train:
        train_data = convert_documents_to_features(
            dataset.train,
            args.tokenizer,
            entity_vocab,
            "train",
            "simple",
            args.max_seq_length,
            args.max_candidate_length,
            args.max_mention_length,
        )
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=collate_fn, shuffle=True)

        logger.info("Fix entity embeddings during training: %s", args.fix_entity_emb)
        if args.fix_entity_emb:
            model.entity_embeddings.entity_embeddings.weight.requires_grad = False
        logger.info("Fix entity bias during training: %s", args.fix_entity_bias)
        if args.fix_entity_bias:
            model.entity_predictions.bias.requires_grad = False

        num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        trainer = EntityDisambiguationTrainer(args, model, train_dataloader, num_train_steps)
        trainer.train()

        if args.output_dir:
            logger.info("Saving model to %s", args.output_dir)
            torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    results = {}

    if args.do_eval:
        model.eval()

        for dataset_name in args.test_set:
            print("***** Dataset: %s *****" % dataset_name)
            eval_documents = getattr(dataset, dataset_name)
            eval_data = convert_documents_to_features(
                eval_documents,
                args.tokenizer,
                entity_vocab,
                "eval",
                args.document_split_mode,
                args.max_seq_length,
                args.max_candidate_length,
                args.max_mention_length,
            )
            eval_dataloader = DataLoader(
                eval_data, batch_size=1, collate_fn=functools.partial(collate_fn, is_eval=True)
            )
            predictions_file = None
            if args.output_dir:
                predictions_file = os.path.join(args.output_dir, "eval_predictions_%s.jsonl" % dataset_name)
            results[dataset_name] = evaluate(args, eval_dataloader, model, entity_vocab, predictions_file)

        if args.output_dir:
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)

    return results


@cli.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.File("w"))
@click.option("--data-dir", type=click.Path(exists=True), default="data/entity-disambiguation")
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
        out_file.write(title + "\n")


@cli.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.File(mode="w"))
def create_title_list(dump_db_file, out_file):
    dump_db = DumpDB(dump_db_file)

    for title in dump_db.titles():
        out_file.write(f"{title}\n")


@cli.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.File(mode="w"))
def create_redirect_tsv(dump_db_file, out_file):
    dump_db = DumpDB(dump_db_file)

    for src, dest in dump_db.redirects():
        out_file.write(f"{src}\t{dest}\n")


class EntityDisambiguationTrainer(Trainer):
    def _create_model_arguments(self, batch):
        batch["entity_labels"] = batch["entity_ids"].clone()
        for index, entity_length in enumerate(batch["entity_attention_mask"].sum(1).tolist()):
            masked_entity_length = max(1, round(entity_length * self.args.masked_entity_prob))
            permutated_indices = torch.randperm(entity_length)[:masked_entity_length]
            batch["entity_ids"][index, permutated_indices[:masked_entity_length]] = 1  # [MASK]
            batch["entity_labels"][index, permutated_indices[masked_entity_length:]] = -1

        return batch


def evaluate(args, eval_dataloader, model, entity_vocab, output_file=None):
    predictions = []
    context_entities = []
    labels = []
    documents = []
    mentions = []
    reverse_entity_vocab = {v: k for k, v in entity_vocab.items()}
    for item in tqdm(eval_dataloader, leave=False):  # the batch size must be 1
        inputs = {
            k: v.to(args.device) for k, v in item.items() if k not in ("document", "mentions", "target_mention_indices")
        }
        entity_ids = inputs.pop("entity_ids")
        entity_attention_mask = inputs.pop("entity_attention_mask")
        input_entity_ids = entity_ids.new_full(entity_ids.size(), 1)  # [MASK]
        entity_length = entity_ids.size(1)
        with torch.no_grad():
            if args.use_context_entities:
                result = torch.zeros(entity_length, dtype=torch.long)
                prediction_order = torch.zeros(entity_length, dtype=torch.long)
                for n in range(entity_length):
                    logits = model(entity_ids=input_entity_ids, entity_attention_mask=entity_attention_mask, **inputs)[
                        0
                    ]
                    probs = F.softmax(logits, dim=2) * (input_entity_ids == 1).unsqueeze(-1).type_as(logits)
                    max_probs, max_indices = torch.max(probs.squeeze(0), dim=1)
                    if args.context_entity_selection_order == "highest_prob":
                        target_index = torch.argmax(max_probs, dim=0)
                    elif args.context_entity_selection_order == "random":
                        target_index = random.choice((input_entity_ids == 1).squeeze(0).nonzero().view(-1).tolist())
                    elif args.context_entity_selection_order == "natural":
                        target_index = (input_entity_ids == 1).squeeze(0).nonzero().view(-1)[0]
                    input_entity_ids[0, target_index] = max_indices[target_index]
                    result[target_index] = max_indices[target_index]
                    prediction_order[target_index] = n
            else:
                logits = model(entity_ids=input_entity_ids, entity_attention_mask=entity_attention_mask, **inputs)[0]
                result = torch.argmax(logits, dim=2).squeeze(0)

        for index in item["target_mention_indices"][0]:
            predictions.append(result[index].item())
            labels.append(entity_ids[0, index].item())
            documents.append(item["document"][0])
            mentions.append(item["mentions"][0][index])
            if args.use_context_entities:
                context_entities.append(
                    [
                        dict(
                            order=prediction_order[n].item(),
                            prediction=reverse_entity_vocab[result[n].item()],
                            label=mention.title,
                            text=mention.text,
                        )
                        for n, mention in enumerate(item["mentions"][0])
                        if prediction_order[n] < prediction_order[index]
                    ]
                )
            else:
                context_entities.append([])

    num_correct = 0
    num_mentions = 0
    num_mentions_with_candidates = 0

    eval_predictions = []
    for prediction, label, document, mention, cxt in zip(predictions, labels, documents, mentions, context_entities):
        if prediction == label:
            num_correct += 1

        assert not (mention.candidates and prediction == 0)
        assert label != 0

        num_mentions += 1
        if mention.candidates:
            num_mentions_with_candidates += 1

            eval_predictions.append(
                dict(
                    document_id=document.id,
                    document_words=document.words,
                    document_length=len(document.words),
                    mention_length=len(document.mentions),
                    mention=dict(
                        label=mention.title,
                        text=mention.text,
                        span=(mention.start, mention.end),
                        candidate_length=len(mention.candidates),
                        candidates=[dict(prior_prob=c.prior_prob, title=c.title) for c in mention.candidates],
                    ),
                    prediction=reverse_entity_vocab[prediction],
                    context_entities=cxt,
                )
            )

    if output_file:
        with open(output_file, "w") as f:
            for obj in eval_predictions:
                f.write(json.dumps(obj) + "\n")

    precision = num_correct / num_mentions_with_candidates
    recall = num_correct / num_mentions
    f1 = 2.0 * precision * recall / (precision + recall)

    print("F1: %.5f" % f1)
    print("Precision: %.5f" % precision)
    print("Recall: %.5f" % recall)

    return dict(precision=precision, recall=recall, f1=f1)
