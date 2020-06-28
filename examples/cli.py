import json
import logging
import os
import random
import string
import subprocess
import sys

logging.getLogger("transformers").setLevel(logging.WARNING)

import click
import comet_ml
import torch
from transformers import AutoTokenizer

from luke.model import LukeConfig
from luke.utils.entity_vocab import EntityVocab
from luke.utils.model_utils import ENTITY_VOCAB_FILE, METADATA_FILE

from .utils.mention_db import MentionDB

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"

try:
    import absl.logging

    # https://github.com/tensorflow/tensorflow/issues/27045#issuecomment-519642980
    logging.getLogger().removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except ImportError:
    pass

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--output-dir",
    default="exp_" + "".join(random.choice(string.ascii_letters) for m in range(8)),
    type=click.Path(),
    required=True,
)
@click.option("--verbose", is_flag=True)
@click.option("--num-gpus", default=1)
@click.option("--master-port", default=29500)
@click.option("--local-rank", "--local_rank", default=-1)
@click.option("--model-dir", type=click.Path(exists=True))
@click.option("--weights-file", type=click.Path(exists=True))
@click.option("--mention-db-file", type=click.Path(exists=True))
@click.option("--comet-project")
@click.option("--comet-offline", is_flag=True)
@click.option("--comet-offline-dir", type=click.Path(), default="comet_experiments")
@click.pass_context
def cli(
    ctx,
    num_gpus,
    master_port,
    output_dir,
    verbose,
    local_rank,
    model_dir,
    weights_file,
    mention_db_file,
    comet_project,
    comet_offline,
    comet_offline_dir,
):
    if local_rank == -1 and num_gpus > 1:
        current_env = os.environ.copy()
        current_env["MASTER_ADDR"] = "127.0.0.1"
        current_env["MASTER_PORT"] = str(master_port)
        current_env["WORLD_SIZE"] = str(num_gpus)

        processes = []

        for local_rank in range(0, num_gpus):
            current_env["RANK"] = str(local_rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            cmd = [sys.executable, "-u", "-m", "examples.cli", "--local-rank={}".format(local_rank)]
            cmd.extend(sys.argv[1:])

            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)

        for process in processes:
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)

        sys.exit(0)

    else:
        if local_rank not in (-1, 0):
            logging.basicConfig(format=LOG_FORMAT, level=logging.WARNING)
        elif verbose:
            logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
        else:
            logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

        if not os.path.exists(output_dir) and local_rank in [-1, 0]:
            os.makedirs(output_dir)
            logger.info("Output dir: %s", output_dir)

        # NOTE: ctx.obj is documented here: http://click.palletsprojects.com/en/7.x/api/#click.Context.obj
        ctx.obj = dict(local_rank=local_rank, output_dir=output_dir)

        if local_rank == -1:
            ctx.obj["device"] = torch.device("cuda")
        else:
            torch.cuda.set_device(local_rank)
            ctx.obj["device"] = torch.device("cuda", local_rank)
            torch.distributed.init_process_group(backend="nccl")

        if "experiment" not in ctx.obj and ctx.obj["local_rank"] in (-1, 0) and comet_project:
            if comet_offline:
                ctx.obj["experiment"] = comet_ml.OfflineExperiment(
                    project_name=comet_project,
                    offline_directory=comet_offline_dir,
                    auto_metric_logging=False,
                    auto_output_logging=None,
                    log_code=False,
                    log_graph=False,
                    log_env_host=False,
                    log_env_gpu=False,
                    log_env_cpu=False,
                )
            else:
                ctx.obj["experiment"] = comet_ml.Experiment(
                    project_name=comet_project,
                    auto_metric_logging=False,
                    auto_output_logging=None,
                    log_code=False,
                    log_graph=False,
                    log_env_host=False,
                    log_env_gpu=False,
                    log_env_cpu=False,
                )
        else:
            ctx.obj["experiment"] = comet_ml.OfflineExperiment(offline_directory=output_dir, disabled=True)

        local_dict = locals()
        ctx.obj["experiment"].log_parameters({p.name: local_dict[p.name] for p in cli.params})

        if model_dir or weights_file:
            if not model_dir:
                model_dir = os.path.dirname(weights_file)
            ctx.obj["model_dir"] = model_dir

            json_file = os.path.join(model_dir, METADATA_FILE)
            with open(json_file) as f:
                model_data = json.load(f)

            # for backward compatibility
            if "entity_emb_size" not in model_data["model_config"]:
                model_data["model_config"]["entity_emb_size"] = model_data["model_config"]["hidden_size"]

            ctx.obj["tokenizer"] = AutoTokenizer.from_pretrained(model_data["model_config"]["bert_model_name"])
            ctx.obj["entity_vocab"] = EntityVocab(os.path.join(model_dir, ENTITY_VOCAB_FILE))
            ctx.obj["bert_model_name"] = model_data["model_config"]["bert_model_name"]
            ctx.obj["model_config"] = LukeConfig(**model_data["model_config"])
            ctx.obj["max_mention_length"] = model_data["max_mention_length"]

        if weights_file:
            ctx.obj["weights_file"] = weights_file
            ctx.obj["model_weights"] = torch.load(weights_file, map_location="cpu")
            ctx.obj["experiment"].log_parameter("weights_file_name", os.path.basename(weights_file))

        if mention_db_file:
            ctx.obj["mention_db"] = MentionDB(mention_db_file)
        else:
            ctx.obj["mention_db"] = None


from .entity_disambiguation.main import cli as entity_disambiguation_cli

cli.add_command(entity_disambiguation_cli)
from .entity_typing.main import cli as entity_typing_cli

cli.add_command(entity_typing_cli)
from .ner.main import cli as ner_cli

cli.add_command(ner_cli)
from .reading_comprehension.main import cli as reading_comprehension_cli

cli.add_command(reading_comprehension_cli)
from .relation_classification.main import cli as relation_classification_cli

cli.add_command(relation_classification_cli)
from .utils.mention_db import cli as mention_db_cli

cli.add_command(mention_db_cli)
from .entity_span_qa.main import cli as entity_span_qa_cli

cli.add_command(entity_span_qa_cli)


if __name__ == "__main__":
    cli()
