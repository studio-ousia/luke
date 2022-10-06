import logging
import os

import click
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from luke.utils.entity_vocab import MASK_TOKEN, EntityVocab

from dataloader import create_dataloader
from dataset import load_dataset
from model import LukeForEntityDisambiguation

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model-dir", type=click.Path(exists=True), required=True)
@click.option("--dataset-dir", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), required=True)
@click.option("--titles-file", type=click.Path(exists=True), required=True)
@click.option("--redirects-file", type=click.Path(exists=True), required=True)
@click.option("--batch-size", type=int, default=16)
@click.option("--gradient-accumulation-steps", type=int, default=1)
@click.option("--learning-rate", type=float, default=2e-5)
@click.option("--num-epochs", type=int, default=2)
@click.option("--warmup-ratio", type=float, default=0.1)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--max-grad-norm", type=float, default=1.0)
@click.option("--masked-entity-prob", type=float, default=0.9)
@click.option("--max-seq-length", type=int, default=512)
@click.option("--max-entity-length", type=int, default=128)
@click.option("--max-candidate-length", type=int, default=30)
@click.option("--max-mention-length", type=int, default=30)
@click.option("--device", type=str, default="cuda")
def train(
    model_dir: str,
    dataset_dir: str,
    output_dir: str,
    titles_file: str,
    redirects_file: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_epochs: int,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    masked_entity_prob: float,
    max_seq_length: int,
    max_entity_length: int,
    max_candidate_length: int,
    max_mention_length: int,
    device: str,
):
    model = LukeForEntityDisambiguation.from_pretrained(model_dir).train()
    model.to(device)
    model.luke.entity_embeddings.entity_embeddings.weight.requires_grad = False
    assert not model.entity_predictions.decoder.weight.requires_grad
    model.entity_predictions.bias.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    entity_vocab_path = os.path.join(model_dir, "entity_vocab.jsonl")
    entity_vocab = EntityVocab(entity_vocab_path)
    mask_entity_id = entity_vocab[MASK_TOKEN]

    dataset = load_dataset(dataset_dir=dataset_dir, titles_file=titles_file, redirects_file=redirects_file,)
    documents = dataset.get_dataset("train")
    dataloader = create_dataloader(
        documents=documents,
        tokenizer=tokenizer,
        entity_vocab=entity_vocab,
        batch_size=batch_size,
        fold="train",
        document_split_mode="simple",
        max_seq_length=max_seq_length,
        max_entity_length=max_entity_length,
        max_candidate_length=max_candidate_length,
        max_mention_length=max_mention_length,
    )
    num_train_steps = len(dataloader) // gradient_accumulation_steps * num_epochs

    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_parameters, lr=learning_rate, eps=1e-6, correct_bias=False)

    warmup_steps = int(num_train_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)

    step = 0
    for epoch_num in range(num_epochs):
        with tqdm(dataloader) as pbar:
            for batch in pbar:
                batch["labels"] = batch["entity_ids"].clone()
                for index, entity_length in enumerate(batch["entity_attention_mask"].sum(1).tolist()):
                    masked_entity_length = max(1, round(entity_length * masked_entity_prob))
                    permutated_indices = torch.randperm(entity_length)[:masked_entity_length]
                    batch["entity_ids"][index, permutated_indices[:masked_entity_length]] = mask_entity_id
                    batch["labels"][index, permutated_indices[masked_entity_length:]] = -1
                batch = {k: v.to(device) for k, v in batch.items() if k != "eval_entity_mask"}
                outputs = model(**batch)
                loss = outputs[0]
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    pbar.set_description(f"epoch: {epoch_num} loss: {loss:.7f}")
                step += 1

    os.makedirs(output_dir, exist_ok=True)
    entity_vocab.save(os.path.join(output_dir, "entity_vocab.jsonl"))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
