# -*- coding: utf-8 -*-

import datetime
import functools
import json
import logging
import multiprocessing
import os
import click
import joblib
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader

logger = logging.getLogger(__name__)


@click.group()
def cli():
    fmt = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=fmt)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--strip-accents/--no-strip-accents', default=True)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', type=int, default=100)
def build_dump_db(dump_file, out_file, strip_accents, **kwargs):
    from utils import clean_text

    dump_reader = WikiDumpReader(dump_file)
    func = clean_text
    if strip_accents:
        func = functools.partial(clean_text, strip_accents=True)

    DumpDB.build(dump_reader, out_file, preprocess_func=func, **kwargs)


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--min-link-prob', default=0.1)
@click.option('--max-candidate-size', default=100)
@click.option('--min-link-count', default=0)
@click.option('--max-mention-len', default=100)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=30)
def build_mention_db(dump_db_file, out_file, **kwargs):
    from utils.entity_linker import MentionDB

    dump_db = DumpDB(dump_db_file)
    mention_db = MentionDB.build(dump_db, **kwargs)
    mention_db.save(out_file)


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('mention_db_file', type=click.Path(exists=True))
@click.argument('word_vocab_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--target', type=click.Choice(['abstract', 'full']), default='full')
@click.option('--uncased/--cased', default=True)
@click.option('--min-prior-prob', default=0.1)
@click.option('--min-sentence-len', default=10)
@click.option('--pool-size', default=multiprocessing.cpu_count())
def build_corpus_data(dump_db_file, mention_db_file, word_vocab_file, uncased, min_prior_prob,
                      **kwargs):
    from utils.vocab import WordPieceVocab
    from utils.word_tokenizer import WordPieceTokenizer
    from utils.sentence_tokenizer import OpenNLPSentenceTokenizer
    from utils.entity_linker import MentionDB, EntityLinker
    from wiki_corpus import WikiCorpus

    dump_db = DumpDB(dump_db_file)
    word_vocab = WordPieceVocab(word_vocab_file)
    tokenizer = WordPieceTokenizer(word_vocab, lowercase=uncased)
    sentence_tokenizer = OpenNLPSentenceTokenizer()

    mention_db = MentionDB.load(mention_db_file)
    entity_linker = EntityLinker(mention_db, min_prior_prob)

    WikiCorpus.build_corpus_data(dump_db, tokenizer, sentence_tokenizer, entity_linker, **kwargs)


@cli.command()
@click.argument('corpus_data_file', type=click.Path())
@click.argument('out_file', type=click.Path())
@click.option('--vocab-size', default=1000000)
@click.option('-i', '--include', type=click.File(), multiple=True)
def build_entity_vocab(corpus_data_file, include, **kwargs):
    from wiki_corpus import WikiCorpus
    from utils.vocab import EntityVocab

    corpus = WikiCorpus(corpus_data_file)
    white_list = [line.rstrip() for f in include for line in f]
    EntityVocab.build_vocab(corpus, white_list=white_list, **kwargs)


def common_training_options(func):
    @click.argument('corpus_data_file', type=click.Path())
    @click.argument('entity_vocab_file', type=click.Path(exists=True))
    @click.option('--run-name', type=click.Path(),
                  default=datetime.datetime.now().strftime('job_%Y%m%d-%H%M%S'))
    @click.option('--base-output-dir', type=click.Path(), default='out')
    @click.option('--base-log-dir', type=click.Path(), default='log')
    @click.option('--mmap', is_flag=True)
    @click.option('--single-sentence/--sentence-pair', is_flag=True)
    @click.option('--single-token-per-mention/--multiple-token-per-mention', default=True)
    @click.option('--batch-size', default=256)  # BERT default=256
    @click.option('--gradient-accumulation-steps', default=1)
    @click.option('--learning-rate', default=1e-4)  # BERT original=1e-4, recommended for fine-tuning: 2e-5
    @click.option('--lr-decay/--no-lr-decay', default=False)
    @click.option('--warmup-steps', default=10000)
    @click.option('--max-seq-length', default=512)  # BERT default=512
    @click.option('--max-entity-length', default=256)
    @click.option('--max-mention-length', default=100)
    @click.option('--short-seq-prob', default=0.1)
    @click.option('--masked-lm-prob', default=0.15)
    @click.option('--max-predictions-per-seq', default=77)  # 512 * 0.15
    @click.option('--num-train-steps', default=300000)
    @click.option('--num-page-chunks', default=100)
    @click.option('--save-every', default=5000)
    @click.option('--entity-emb-size', default=768)
    @click.option('--bert-model-name', default='bert-base-uncased')
    @click.option('--model-file', type=click.Path(exists=True), default=None)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def run_training_options(func):
    @click.option('--masked-entity-prob', default=0.15)
    @click.option('--max-entity-predictions-per-seq', default=38)  # 256 * 0.15
    @click.option('--update-all-weights', is_flag=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def run_e2e_training_options(func):
    @click.option('--link-prob-bin-size', default=20)
    @click.option('--prior-prob-bin-size', default=20)
    @click.option('--entity-classification/--no-entity-classification', default=True)
    @click.option('--pretrained-model-file', type=click.Path(exists=True), default=None)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@common_training_options
@run_training_options
@click.option('-j', '--json-data', default=None)
def run_training(json_data, **kwargs):
    import train
    _run_training(train.run_training, json_data, **kwargs)


@cli.command()
@common_training_options
@run_e2e_training_options
@click.option('-j', '--json-data', default=None)
def run_e2e_training(json_data, **kwargs):
    import train
    _run_training(train.run_e2e_training, json_data, **kwargs)


def _run_training(train_func, json_data, **kwargs):
    if json_data is not None:
        kwargs.update(json.loads(json_data))

    run_name = kwargs.pop('run_name')
    output_dir = os.path.join(kwargs.pop('base_output_dir'), run_name)
    kwargs['output_dir'] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(kwargs.pop('base_log_dir'), run_name)
    kwargs['log_dir'] = log_dir
    os.makedirs(log_dir, exist_ok=True)

    train_func(**kwargs)


def resume_training_options(func):
    @click.argument('output_dir', type=click.Path())
    @click.option('--global-step', default=None, type=int)
    @click.option('--batch-size', default=None, type=int)
    @click.option('--gradient-accumulation-steps', default=None, type=int)
    @click.option('--learning-rate', default=None, type=float)
    @click.option('--lr-decay/--no-lr-decay', default=None)
    @click.option('--num-train-steps', default=None, type=int)
    @click.option('--save-every', default=None, type=int)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@resume_training_options
@click.option('-j', '--json-data', default=None)
def resume_training(json_data, **kwargs):
    import train
    _resume_training(train.run_training, json_data, **kwargs)


@cli.command()
@resume_training_options
@click.option('-j', '--json-data', default=None)
def resume_e2e_training(json_data, **kwargs):
    import train
    _resume_training(train.run_e2e_training, json_data, **kwargs)


def _resume_training(train_func, json_data, output_dir, global_step, **kwargs):
    if json_data is not None:
        kwargs.update(json.loads(json_data))

    if global_step is None:
        # get the latest data file
        data_file = sorted([f for f in os.listdir(output_dir) if f.startswith('data_')])[-1]
        global_step = int(data_file.replace('data_step', '').replace('.pkl', ''))
    else:
        data_file = 'data_step%07d.pkl' % global_step

    data = joblib.load(os.path.join(output_dir, data_file))

    args = data['args']
    args['global_step'] = data['global_step']
    args['page_chunks'] = data['page_chunks']

    model_file = data_file.replace('.pkl', '.bin').replace('data', 'model')
    args['model_file'] = os.path.join(output_dir, model_file)
    optimizer_file = data_file.replace('.pkl', '.bin').replace('data', 'optimizer')
    args['optimizer_file'] = os.path.join(output_dir, optimizer_file)
    sparse_optimizer_file = data_file.replace('.pkl', '.bin').replace('data', 'sparse_optimizer')
    if os.path.exists(sparse_optimizer_file):
        args['sparse_optimizer_file'] = os.path.join(output_dir, sparse_optimizer_file)

    for (key, value) in kwargs.items():
        if value is not None:
            args[key] = value

    train_func(**args)


def task_common_options(func):
    @click.argument('model_file', type=click.Path(exists=True))
    @click.option('--output-dir', type=click.Path())
    @click.option('--data-dir', type=click.Path(exists=True), default='data')
    @click.option('--max-seq-length', default=512)
    @click.option('--batch-size', default=32)
    @click.option('--learning-rate', default=3e-5)
    @click.option('--iteration', default=4.0)
    @click.option('--eval-batch-size', default=8)
    @click.option('--warmup-proportion', default=0.1)
    @click.option('--lr-decay/--no-lr-decay', default=True)
    @click.option('--seed', default=42)
    @click.option('--gradient-accumulation-steps', default=1)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.option('--max-entity-length', default=128)
@click.option('--max-candidate-size', default=30)
@click.option('--min-context-prior-prob', default=0.9)
@click.option('--fix-entity-emb/--update-entity-emb', default=True)
@click.option('-t', '--test-set', default=['test_b'], multiple=True)
@task_common_options
def entity_disambiguation(dump_db_file, data_dir, **kwargs):
    from entity_disambiguation.main import run

    dump_db = DumpDB(dump_db_file)
    data_dir = os.path.join(data_dir, 'entity-disambiguation')

    run(data_dir, dump_db, **kwargs)


def glue_options(func):
    @task_common_options
    @click.option('-t', '--task-name', default='mrpc', type=click.Choice(['cola', 'mnli', 'qnli',
        'mrpc', 'rte', 'qqp', 'scitail', 'sts-b']))
    @click.option('--max-entity-length', default=128)
    @click.option('--min-prior-prob', default=[0.1, 0.3, 0.9], type=float, multiple=True)
    @click.option('--fix-entity-emb/--update-entity-emb', default=[True, False], multiple=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def multi_choice_qa_options(func):
    @task_common_options
    @click.option('-t', '--task-name', type=click.Choice(['swag', 'arc-easy', 'arc-challenge',
        'openbookqa']), default='swag')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@glue_options
@click.option('-j', '--json-data', default=None)
def glue(**kwargs):
    from glue import run
    run_task(run, **kwargs)


@cli.command()
@multi_choice_qa_options
@click.option('-j', '--json-data', default=None)
def multi_choice_qa(**kwargs):
    from multi_choice_qa import run
    run_task(run, **kwargs)


def run_task(run_func, word_vocab_file, mention_db_file, json_data, **kwargs):
    from utils.vocab import WordPieceVocab
    from utils.word_tokenizer import WordPieceTokenizer
    from utils.entity_linker import MentionDB, EntityLinker

    if json_data is not None:
        kwargs.update(json.loads(json_data))

    word_vocab = WordPieceVocab(word_vocab_file)
    uncased = kwargs.pop('uncased')
    tokenizer = WordPieceTokenizer(word_vocab, lowercase=uncased)

    mention_db = MentionDB.load(mention_db_file)
    entity_linker = EntityLinker(mention_db)

    learning_rates = kwargs.pop('learning_rate')
    iterations = kwargs.pop('iteration')
    batch_sizes = kwargs.pop('batch_size')
    min_prior_probs = kwargs.pop('min_prior_prob')
    fix_entity_embs = kwargs.pop('fix_entity_emb')
    for learing_rate in learning_rates:
        for iteration in iterations:
            for batch_size in batch_sizes:
                for min_prior_prob in min_prior_probs:
                    for fix_entity_emb in fix_entity_embs:
                        run_func(tokenizer, entity_linker, learning_rate=learing_rate,
                                 iteration=iteration, min_prior_prob=min_prior_prob,
                                 batch_size=batch_size, fix_entity_emb=fix_entity_emb, **kwargs)


if __name__ == '__main__':
    cli()
