import json
import logging
import os
import random
import string

logging.getLogger('transformers').setLevel(logging.WARNING)

import click
import comet_ml
import torch
from transformers import AutoTokenizer

from luke.model import LukeConfig
from luke.utils.entity_vocab import EntityVocab
from luke.pretraining.dataset import ENTITY_VOCAB_FILE, METADATA_FILE

from .utils import set_seed
from .utils.mention_db import MentionDB

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'

try:
    import absl.logging
    # https://github.com/tensorflow/tensorflow/issues/27045#issuecomment-519642980
    logging.getLogger().removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except ImportError:
    pass

logger = logging.getLogger(__name__)


@click.group()
@click.option('-o', '--output-dir', default='exp_' + ''.join(random.choice(string.ascii_letters) for m in range(8)),
              type=click.Path(), required=True)
@click.option('-v', '--verbose', is_flag=True)
@click.option('--seed', default=1)
@click.option('--no-cuda', is_flag=True)
@click.option('--local-rank', '--local_rank', default=-1)
@click.option('--model-dir', type=click.Path(exists=True))
@click.option('--weights-file', type=click.Path(exists=True))
@click.option('--mention-db-file', type=click.Path(exists=True))
@click.pass_context
def cli(ctx, output_dir, verbose, seed, no_cuda, local_rank, model_dir, weights_file, mention_db_file):
    if local_rank not in [-1, 0]:
        logging.basicConfig(format=LOG_FORMAT, level=logging.WARNING)
    elif verbose:
        logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

    if not os.path.exists(output_dir) and local_rank in [-1, 0]:
        os.makedirs(output_dir)

    # NOTE: ctx.obj is documented here: http://click.palletsprojects.com/en/7.x/api/#click.Context.obj
    ctx.obj = dict(local_rank=local_rank, output_dir=output_dir)

    if no_cuda or not torch.cuda.is_available():
        ctx.obj['device'] = torch.device('cpu')
    elif local_rank == -1:
        ctx.obj['device'] = torch.device('cuda')
    else:
        torch.cuda.set_device(local_rank)
        ctx.obj['device'] = torch.device('cuda', local_rank)
        torch.distributed.init_process_group(backend='nccl')

    if model_dir or weights_file:
        if not model_dir:
            model_dir = os.path.dirname(weights_file)
        ctx.obj['model_dir'] = model_dir

        json_file = os.path.join(model_dir, METADATA_FILE)
        with open(json_file) as f:
            model_data = json.load(f)

        ctx.obj['tokenizer'] = AutoTokenizer.from_pretrained(model_data['model_config']['bert_model_name'])
        ctx.obj['entity_vocab'] = EntityVocab(os.path.join(model_dir, ENTITY_VOCAB_FILE))
        ctx.obj['model_config'] = LukeConfig(**model_data['model_config'])
        ctx.obj['max_mention_length'] = model_data['max_mention_length']

    if weights_file:
        ctx.obj['model_weights'] = torch.load(weights_file, map_location='cpu')

    if mention_db_file:
        ctx.obj['mention_db'] = MentionDB(mention_db_file)

    set_seed(seed)


from .entity_disambiguation.main import cli as entity_disambiguation_cli
cli.add_command(entity_disambiguation_cli)
from .squad.main import cli as squad_cli
cli.add_command(squad_cli)
from .utils.mention_db import cli as mention_db_cli
cli.add_command(mention_db_cli)


if __name__ == '__main__':
    cli()
