import logging
import os
import random
import string
import subprocess
import sys
from argparse import Namespace

logging.getLogger("transformers").setLevel(logging.WARNING)

import click
import torch

from luke.utils.model_utils import ModelArchive

from examples.legacy.utils.experiment_logger import commet_logger_args, CometLogger, NullLogger

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
    "--output-dir", default="exp_" + "".join(random.choice(string.ascii_letters) for m in range(8)), type=click.Path()
)
@click.option("--num-gpus", default=1)
@click.option("--experiment-logger", "--logger", type=click.Choice(["comet"]))
@click.option("--master-port", default=29500)
@click.option("--local-rank", "--local_rank", default=-1)
@click.option("--model-file", type=click.Path(exists=True))
@commet_logger_args
@click.pass_context
def cli(ctx, **kwargs):
    args = Namespace(**kwargs)

    if args.local_rank == -1 and args.num_gpus > 1:
        current_env = os.environ.copy()
        current_env["MASTER_ADDR"] = "127.0.0.1"
        current_env["MASTER_PORT"] = str(args.master_port)
        current_env["WORLD_SIZE"] = str(args.num_gpus)

        processes = []

        for args.local_rank in range(0, args.num_gpus):
            current_env["RANK"] = str(args.local_rank)
            current_env["LOCAL_RANK"] = str(args.local_rank)

            cmd = [sys.executable, "-u", "-m", "examples.cli", "--local-rank={}".format(args.local_rank)]
            cmd.extend(sys.argv[1:])

            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)

        for process in processes:
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)

        sys.exit(0)
    else:
        if args.local_rank not in (-1, 0):
            logging.basicConfig(format=LOG_FORMAT, level=logging.WARNING)
        else:
            logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
            logger.info("Output dir: %s", args.output_dir)

        # NOTE: ctx.obj is documented here: http://click.palletsprojects.com/en/7.x/api/#click.Context.obj
        ctx.obj = dict(local_rank=args.local_rank, output_dir=args.output_dir)

        if args.num_gpus == 0:
            ctx.obj["device"] = torch.device("cpu")
        elif args.local_rank == -1:
            ctx.obj["device"] = torch.device("cuda")
        else:
            torch.cuda.set_device(args.local_rank)
            ctx.obj["device"] = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")

        experiment_logger = NullLogger()

        if args.local_rank in (-1, 0) and args.experiment_logger == "comet":
            experiment_logger = CometLogger(args)

        experiment_logger.log_parameters({p.name: getattr(args, p.name) for p in cli.params})
        ctx.obj["experiment"] = experiment_logger

        if args.model_file:
            model_archive = ModelArchive.load(args.model_file)
            ctx.obj["tokenizer"] = model_archive.tokenizer
            ctx.obj["entity_vocab"] = model_archive.entity_vocab
            ctx.obj["bert_model_name"] = model_archive.bert_model_name
            ctx.obj["model_config"] = model_archive.config
            ctx.obj["max_mention_length"] = model_archive.max_mention_length
            ctx.obj["model_weights"] = model_archive.state_dict

            experiment_logger.log_parameter("model_file_name", os.path.basename(args.model_file))


from examples.entity_disambiguation.main import cli as entity_disambiguation_cli

cli.add_command(entity_disambiguation_cli)


if __name__ == "__main__":
    cli()
