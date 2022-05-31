import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import click
import torch
import tqdm
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.nn import util as nn_util
from transformers import AutoModelForMaskedLM

from examples.mlama.reader import MultilingualLAMAReader


@click.command()
@click.argument("mlama-path", type=click.Path(exists=True))
@click.argument("language", type=str)
@click.argument("transformers-model-name")
@click.argument("output-file-path", type=str)
@click.option("--batch-size", type=int, default=8)
@click.option("--cuda-device", default=-1)
@click.option("--max-instances", type=int, default=None)
@click.option("--use-subject-entity-mask", is_flag=True)
@click.option("--use-subject-entity", is_flag=True)
@click.option("--use-object-entity", is_flag=True)
@click.option("--entity-vocab-file", type=click.Path(exists=True))
@click.option("--num-workers", type=int, default=0)
@click.option("--max-instances-in-memory", type=int, default=None)
@torch.no_grad()
def evaluate_mlama(
    mlama_path: str,
    language: str,
    transformers_model_name: str,
    output_file_path: str,
    batch_size: int,
    cuda_device: int,
    max_instances: int,
    use_subject_entity_mask: bool,
    use_subject_entity: bool,
    use_object_entity: bool,
    entity_vocab_file: str,
    num_workers: int,
    max_instances_in_memory: int,
):

    model = AutoModelForMaskedLM.from_pretrained(transformers_model_name)

    if cuda_device > -1:
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    data_loader = MultiProcessDataLoader(
        reader=MultilingualLAMAReader(
            mlama_path=mlama_path,
            transformers_model_name=transformers_model_name,
            max_instances=max_instances,
            use_subject_entity_mask=use_subject_entity_mask,
            use_subject_entity=use_subject_entity,
            use_object_entity=use_object_entity,
            entity_vocab_path=entity_vocab_file,
        ),
        data_path=language,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_instances_in_memory=max_instances_in_memory,
    )

    vocab = Vocabulary.from_pretrained_transformer(transformers_model_name)
    data_loader.index_with(vocab)

    results: Dict[str, Dict] = defaultdict(lambda: {"answer": None, "candidates": []})
    for batch in tqdm.tqdm(data_loader):
        batch = nn_util.move_to_device(batch, device)

        model_input = dict(
            input_ids=batch["input_tokens"]["tokens"]["token_ids"],
            attention_mask=batch["input_tokens"]["tokens"]["mask"],
            token_type_ids=batch["input_tokens"]["tokens"]["type_ids"],
        )
        if "entity_ids" in batch:
            model_input.update({k: v for k, v in batch.items() if "entity" in k})

        output = model(**model_input)
        batch_log_probs = torch.log_softmax(output.logits, dim=-1)
        if use_object_entity:
            batch_entity_log_probs = torch.log_softmax(output.entity_logits, dim=-1)
            if use_subject_entity or use_subject_entity_mask:
                batch_object_entity_log_probs = batch_entity_log_probs[:, 1]
            else:
                batch_object_entity_log_probs = batch_entity_log_probs[:, 0]

        for i, (log_probs, masked_span, correct_object, candidate_objects, question, template) in enumerate(
            zip(
                batch_log_probs,
                batch["masked_span"],
                batch["correct_object"],
                batch["candidate_objects"],
                batch["question"],
                batch["template"],
            )
        ):
            mask_log_probs = log_probs[masked_span]
            results[question]["answer"] = correct_object["object"]
            results[question]["template"] = template
            for candidate in candidate_objects:
                score = mask_log_probs[range(len(candidate["ids"])), candidate["ids"]].mean().item()
                if use_object_entity:
                    if candidate["entity_id"] is not None:
                        entity_score = batch_object_entity_log_probs[i][candidate["entity_id"]].item()
                    else:
                        entity_score = None
                    results[question]["candidates"].append((candidate["object"], score, entity_score))
                else:
                    results[question]["candidates"].append((candidate["object"], score))

    ks = [1, 5, 10, 100]
    metrics = {f"k{k}": 0 for k in ks}
    for k in ks:
        num_total = 0
        num_correct = 0
        for result in results.values():
            result["candidates"].sort(key=lambda x: -x[1])
            candidates = {obj for obj, *_ in result["candidates"][:k]}
            num_correct += int(result["answer"] in candidates)
            num_total += 1
        metrics[f"k{k}"] = num_correct / num_total

    Path(output_file_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_file_path, "w") as f:
        json.dump({"metrics": metrics, "output": results}, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    evaluate_mlama()
