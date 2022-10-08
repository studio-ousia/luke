import click
import logging
import os
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from luke.utils.entity_vocab import EntityVocab, MASK_TOKEN, PAD_TOKEN

from dataloader import create_dataloader
from dataset import load_dataset
from model import LukeForEntityDisambiguation

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model-dir", type=click.Path(exists=True), required=True)
@click.option("--dataset-dir", type=click.Path(exists=True), required=True)
@click.option("--titles-file", type=click.Path(exists=True), required=True)
@click.option("--redirects-file", type=click.Path(exists=True), required=True)
@click.option("--ppr-for-ned-dir", type=click.Path(exists=True))
@click.option(
    "-t",
    "--test-set",
    default=["test_b", "ace2004", "aquaint", "msnbc", "wikipedia", "clueweb"],
    type=click.Choice(["test_b", "test_b_ppr", "ace2004", "aquaint", "msnbc", "wikipedia", "clueweb"]),
    multiple=True,
)
@click.option("--device", type=str, default="cuda")
@click.option("--max-seq-length", type=int, default=512)
@click.option("--max-entity-length", type=int, default=128)
@click.option("--max-candidate-length", type=int, default=30)
@click.option("--max-mention-length", type=int, default=30)
@click.option(
    "--inference-mode", type=click.Choice(["global", "local"]), default="global",
)
@click.option(
    "--document-split-mode", type=click.Choice(["simple", "per_mention"]), default="simple",
)
def evaluate(
    model_dir: str,
    dataset_dir: str,
    titles_file: str,
    redirects_file: str,
    ppr_for_ned_dir: Optional[str],
    test_set: List[str],
    device: str,
    max_seq_length: int,
    max_entity_length: int,
    max_candidate_length: int,
    max_mention_length: int,
    inference_mode: str,
    document_split_mode: str,
):
    model = LukeForEntityDisambiguation.from_pretrained(model_dir).eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    entity_vocab_path = os.path.join(model_dir, "entity_vocab.jsonl")
    entity_vocab = EntityVocab(entity_vocab_path)
    pad_entity_id = entity_vocab[PAD_TOKEN]
    mask_entity_id = entity_vocab[MASK_TOKEN]
    dataset = load_dataset(
        dataset_dir=dataset_dir,
        titles_file=titles_file,
        redirects_file=redirects_file,
        ppr_for_ned_dir=ppr_for_ned_dir,
    )

    for dataset_name in test_set:
        print(f"========== Dataset: {dataset_name} ==========")
        documents = dataset.get_dataset(dataset_name)
        dataloader = create_dataloader(
            documents=documents,
            tokenizer=tokenizer,
            entity_vocab=entity_vocab,
            batch_size=1,
            fold="eval",
            document_split_mode=document_split_mode,
            max_seq_length=max_seq_length,
            max_entity_length=max_entity_length,
            max_candidate_length=max_candidate_length,
            max_mention_length=max_mention_length,
        )
        candidate_indices_list = []
        eval_entity_mask_list = []
        for input_dict in tqdm(dataloader, leave=False):
            inputs = {k: v.to(device) for k, v in input_dict.items()}
            entity_ids = inputs.pop("entity_ids")
            entity_length = inputs["entity_attention_mask"].sum()
            input_entity_ids = entity_ids.new_full(entity_ids.size(), pad_entity_id)
            input_entity_ids[0, :entity_length] = mask_entity_id
            eval_entity_mask = inputs.pop("eval_entity_mask")
            eval_entity_mask_list.append(eval_entity_mask[0, :entity_length])
            with torch.no_grad():
                candidate_indices = torch.zeros(entity_length, dtype=torch.long, device=device)
                if inference_mode == "local":
                    logits = model(entity_ids=input_entity_ids, **inputs)[0]
                    for n, entity_id in enumerate(torch.argmax(logits, dim=2)[0, :entity_length]):
                        if inputs["entity_candidate_ids"][0, n].sum() != 0:
                            candidate_indices[n] = (inputs["entity_candidate_ids"][0, n] == entity_id).nonzero(
                                as_tuple=True
                            )[0][0]
                else:
                    for _ in range(entity_length):
                        logits = model(entity_ids=input_entity_ids, **inputs)[0]
                        probs = torch.nn.functional.softmax(logits, dim=2) * (
                            input_entity_ids == mask_entity_id
                        ).unsqueeze(-1).type_as(logits)
                        max_probs, max_indices = torch.max(probs.squeeze(0), dim=1)
                        target_index = torch.argmax(max_probs, dim=0)
                        input_entity_ids[0, target_index] = max_indices[target_index]
                        if inputs["entity_candidate_ids"][0, target_index].sum() != 0:
                            candidate_indices[target_index] = (
                                inputs["entity_candidate_ids"][0, target_index] == max_indices[target_index]
                            ).nonzero(as_tuple=True)[0][0]
            candidate_indices_list.append(candidate_indices)

        all_candidate_indices = torch.cat(candidate_indices_list)
        all_eval_entity_mask = torch.cat(eval_entity_mask_list)

        last_index = -1
        num_correct = 0
        num_mentions = 0
        num_mentions_with_candidates = 0
        for document in documents:
            for mention in document.mentions:
                num_mentions += 1
                index = last_index + 1
                while True:
                    if all_eval_entity_mask[index] == 1:
                        break
                    index += 1
                last_index = index

                if mention.candidates:
                    num_mentions_with_candidates += 1
                    predicted_candidate_index = all_candidate_indices[index]
                    predicted_title = mention.candidates[predicted_candidate_index].title
                    if predicted_title == mention.title:
                        num_correct += 1

        precision = num_correct / num_mentions_with_candidates
        recall = num_correct / num_mentions
        f1 = 2.0 * precision * recall / (precision + recall)
        print(f"F1: {f1:.3f}  Precision: {precision:.3f}  Recall: {recall:.3f}")


if __name__ == "__main__":
    evaluate()
