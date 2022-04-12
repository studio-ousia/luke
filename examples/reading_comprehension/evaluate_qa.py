import json
import os
from pathlib import Path
import logging

import click
import torch

logger = logging.getLogger(__name__)


@click.command()
@click.argument("serialization_dir", type=click.Path(exists=True))
@click.argument("test-file-path", type=str)
@click.option("--output-directory", type=click.Path(), default=None)
@click.option("--mention-candidate-files", type=str)
@click.option("--cuda-device", type=int, default=-1)
@torch.no_grad()
def evaluate_qa(
    serialization_dir: str, test_file_path: str, output_directory: str, mention_candidate_files: str, cuda_device: int
):

    config = json.load(open(Path(serialization_dir) / "config.json"))

    if config["dataset_reader"]["type"] != "transformers_squad":
        raise NotImplementedError("Incorrect serialization directory.")

    transformer_model_name = config["dataset_reader"]["transformer_model_name"]

    if output_directory is None:
        output_directory = Path(serialization_dir) / "evaluation" / Path(test_file_path).stem
        output_directory.mkdir(parents=True, exist_ok=True)

    overrides = {
        "model.qa_metric": {
            "type": "squad-v1.1",
            "gold_data_path": test_file_path,
            "prediction_dump_path": str(Path(output_directory) / "prediction.json"),
            "transformers_tokenizer_name": transformer_model_name,
        }
    }

    if mention_candidate_files:
        mention_candidate_files = json.loads(mention_candidate_files)
        overrides["dataset_reader.wiki_entity_linker.mention_candidate_json_file_paths"] = mention_candidate_files

    output_metric_file = str(Path(output_directory) / "metrics.json")
    command = (
        f"allennlp evaluate {serialization_dir} '{test_file_path}' "
        f"--output-file {output_metric_file} "
        f"--cuda-device {cuda_device} "
        f"-o '{json.dumps(overrides)}' "
        f"--include-package examples"
    )

    os.system(command)
    logger.info(f"Saved the result to {output_directory}")


if __name__ == "__main__":
    evaluate_qa()
