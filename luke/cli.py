import logging
import multiprocessing
import os
import random
import click
import numpy as np
import torch
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # filter out INFO messages from Tensordflow
try:
    import tensorflow
    import absl.logging

    # https://github.com/tensorflow/tensorflow/issues/27045#issuecomment-519642980
    logging.getLogger().removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    # https://github.com/tensorflow/tensorflow/issues/27023#issuecomment-501419334
    from tensorflow.python.util import deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False
except ImportError:
    pass

import luke.pretraining.dataset
import luke.pretraining.train
import luke.utils.entity_vocab


@click.group()
@click.option("--verbose", is_flag=True)
@click.option("--seed", type=int, default=None)
def cli(verbose: bool, seed: int):
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
        logging.getLogger("transformers").setLevel(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)
        logging.getLogger("transformers").setLevel(level=logging.WARNING)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@cli.command()
@click.argument("dump_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", type=int, default=100)
def build_dump_db(dump_file: str, out_file: str, **kwargs):
    dump_reader = WikiDumpReader(dump_file)
    DumpDB.build(dump_reader, out_file, **kwargs)


cli.add_command(luke.utils.entity_vocab.build_entity_vocab)
cli.add_command(luke.pretraining.dataset.build_wikipedia_pretraining_dataset)
cli.add_command(luke.pretraining.train.pretrain)
cli.add_command(luke.pretraining.train.resume_pretraining)
cli.add_command(luke.pretraining.train.start_pretraining_worker)
