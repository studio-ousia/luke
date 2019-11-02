import logging

import click
import comet_ml
import torch

from .utils import set_seed

try:
    import absl.logging
    # https://github.com/tensorflow/tensorflow/issues/27045#issuecomment-519642980
    logging.getLogger().removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except ImportError:
    pass

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'


@click.group()
@click.option('-v', '--verbose', is_flag=True)
@click.option('--seed', default=1)
def cli(verbose, seed):
    if verbose:
        logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

    set_seed(seed)


from .entity_disambiguation.main import cli as entity_disambiguation_cli
cli.add_command(entity_disambiguation_cli)
from .squad.main import cli as squad_cli
cli.add_command(squad_cli)
from .mention_db import cli as mention_db_cli
cli.add_command(mention_db_cli)


if __name__ == '__main__':
    cli()