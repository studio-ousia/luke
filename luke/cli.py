import functools
import json
import logging
import multiprocessing
import os
import random
import subprocess
import tempfile
import click
import numpy as np
import torch
from pytorch_transformers.tokenization_bert import BasicTokenizer, BertTokenizer
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader

logger = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbose', is_flag=True)
@click.option('--seed', default=None)
def cli(verbose, seed):
    fmt = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', type=int, default=100)
def build_dump_db(dump_file, out_file, **kwargs):
    dump_reader = WikiDumpReader(dump_file)
    DumpDB.build(dump_reader, out_file, **kwargs)


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--min-link-prob', default=0.1)
@click.option('--max-candidate-size', default=100)
@click.option('--min-link-count', default=1)
@click.option('--max-mention-length', default=20)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=100)
def build_entity_linker_from_wikipedia(dump_db_file, **kwargs):
    from luke.utils.entity_linker import EntityLinker, BertLowercaseNormalizer

    dump_db = DumpDB(dump_db_file)
    tokenizer = BasicTokenizer(do_lower_case=False)
    normalizer = BertLowercaseNormalizer()
    EntityLinker.build_from_wikipedia(dump_db, tokenizer, normalizer, **kwargs)


@cli.command()
@click.argument('p_e_m_file', type=click.Path(exists=True))
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('wiki_entity_linker_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--max-mention-length', default=20)
def build_entity_linker_from_p_e_m_file(p_e_m_file, dump_db_file, wiki_entity_linker_file, **kwargs):
    from luke.utils.entity_linker import EntityLinker, BertLowercaseNormalizer

    dump_db = DumpDB(dump_db_file)
    tokenizer = BasicTokenizer(do_lower_case=False)
    normalizer = BertLowercaseNormalizer()
    wiki_entity_linker = EntityLinker(wiki_entity_linker_file)
    EntityLinker.build_from_p_e_m_file(p_e_m_file, dump_db, wiki_entity_linker, tokenizer, normalizer, **kwargs)


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('word_vocab_file', type=click.Path(exists=True))
@click.argument('entity_linker_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--cased/--uncased', default=False)
@click.option('--min-sentence-length', default=5)
@click.option('--max-candidate-size', default=10)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=100)
def build_wiki_corpus(dump_db_file, word_vocab_file, entity_linker_file, cased, **kwargs):
    from luke.utils.entity_linker import EntityLinker
    from luke.utils.sentence_tokenizer import OpenNLPSentenceTokenizer
    from luke.utils.wiki_corpus import WikiCorpus

    dump_db = DumpDB(dump_db_file)
    tokenizer = BertTokenizer(word_vocab_file, do_lower_case=not cased)
    sentence_tokenizer = OpenNLPSentenceTokenizer()
    entity_linker = EntityLinker(entity_linker_file)
    WikiCorpus.build(dump_db, tokenizer, sentence_tokenizer, entity_linker, **kwargs)


@cli.command()
@click.argument('corpus_file', type=click.Path())
@click.argument('out_file', type=click.Path())
@click.option('--vocab-size', default=1000000)
@click.option('-w', '--white-list', type=click.File(), multiple=True)
@click.option('--white-list-only', is_flag=True)
def build_entity_vocab(corpus_file, white_list, **kwargs):
    from luke.utils.entity_vocab import EntityVocab
    from luke.utils.wiki_corpus import WikiCorpus

    corpus = WikiCorpus(corpus_file)
    white_list = [line.rstrip() for f in white_list for line in f]
    EntityVocab.build(corpus, white_list=white_list, **kwargs)


def common_train_options(func):
    @functools.wraps(func)
    @click.argument('corpus_file', type=click.Path())
    @click.argument('entity_vocab_file', type=click.Path(exists=True))
    @click.argument('output_dir', type=click.Path())
    @click.option('--parallel', is_flag=True)
    @click.option('--bert-model-name', default='bert-base-uncased')
    @click.option('--single-sentence', is_flag=True)
    @click.option('--max-seq-length', default=512)
    @click.option('--max-entity-length', default=128)
    @click.option('--max-mention-length', default=30)
    @click.option('--short-seq-prob', default=0.0)
    @click.option('--masked-lm-prob', default=0.15)
    @click.option('--masked-entity-prob', default=0.3)
    @click.option('--whole-word-masking', is_flag=True)
    @click.option('--batch-size', default=256)
    @click.option('--gradient-accumulation-steps', default=1)
    @click.option('--learning-rate', default=1e-4)
    @click.option('--lr-schedule', type=click.Choice(['none', 'warmup_constant', 'warmup_linear']),
                  default='warmup_linear')
    @click.option('--warmup-steps', default=0)
    @click.option('--fix-bert-weights', is_flag=True)
    @click.option('--grad-avg-on-cpu', is_flag=True)
    @click.option('--num-train-steps', default=300000)
    @click.option('--num-page-chunks', default=100)
    @click.option('--fp16', is_flag=True)
    @click.option('--fp16-opt-level', default='O1', type=click.Choice(['O0', 'O1', 'O2', 'O3']))
    @click.option('--local-rank', '--local_rank', default=-1)
    @click.option('--log-dir', type=click.Path(), default=None)
    @click.option('--model-file', type=click.Path(exists=True), default=None)
    @click.option('--optimizer-file', type=click.Path(exists=True), default=None)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@common_train_options
def run_training(parallel, **kwargs):
    from luke.pretraining import train
    if parallel:
        run_parallel_training('train.run_training', **kwargs)
    else:
        train.run_training(**kwargs)


@cli.command()
@click.option('--fix-word-emb', is_flag=True)
@click.option('--max-candidate-length', default=10)
@click.option('--min-candidate-prior-prob', default=0.01)
@click.option('--num-el-hidden-layers', default=3)
@click.option('--entity-selector-softmax-temp', default=0.1)
@common_train_options
def run_e2e_training(parallel, **kwargs):
    from luke.pretraining import train
    if parallel:
        run_parallel_training('train.run_e2e_training', **kwargs)
    else:
        train.run_e2e_training(**kwargs)


@cli.command()
@click.argument('func_name')
@click.option('--local-rank', type=int)
@click.option('--kwargs', default=None)
def run_training_function(func_name, local_rank, kwargs):
    from luke.pretraining import train
    if kwargs:
        kwargs = json.loads(kwargs)
    else:
        kwargs = {}
    kwargs['local_rank'] = local_rank
    if 'page_chunks' in kwargs:
        kwargs['page_chunks'] = np.load(kwargs['page_chunks'])

    eval(func_name)(**kwargs)


def run_parallel_training(func_name, **kwargs):
    world_size = torch.cuda.device_count()
    current_env = os.environ.copy()
    current_env['NCCL_SOCKET_IFNAME'] = 'lo'
    current_env['MASTER_ADDR'] = '127.0.0.1'
    current_env['MASTER_PORT'] = '29502'
    current_env['WORLD_SIZE'] = str(world_size)
    current_env['OMP_NUM_THREADS'] = str(1)
    processes = []
    with tempfile.NamedTemporaryFile() as page_chunks_file:
        if 'page_chunks' in kwargs:
            kwargs['page_chunks'] = np.save(page_chunks_file, kwargs['page_chunks'])

        for local_rank in range(world_size):
            cmd = ['luke', 'run-training-function', func_name, f'--local-rank={local_rank}',
                f'--kwargs={json.dumps(kwargs)}']
            current_env['RANK'] = str(local_rank)
            current_env['LOCAL_RANK'] = str(local_rank)
            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)

    try:
        for process in processes:
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()