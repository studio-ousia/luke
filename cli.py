# -*- coding: utf-8 -*-

import datetime
import functools
import json
import logging
import multiprocessing
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
@click.option('--vocab-size', default=1000000)
@click.option('--target', type=click.Choice(['abstract', 'full']), default='full')
@click.option('-i', '--include', type=click.File(), multiple=True)
def build_entity_vocab(dump_db_file, include, **kwargs):
    from utils.vocab import EntityVocab

    dump_db = DumpDB(dump_db_file)
    white_list = [line.rstrip() for f in include for line in f]
    EntityVocab.build_vocab(dump_db, white_list=white_list, **kwargs)


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--min-link-prob', default=0.1)
@click.option('--max-candidate-size', default=100)
@click.option('--min-link-count', default=5)
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
@click.option('--min-prior-prob', default=0.0)
@click.option('--min-sentence-len', default=10)
@click.option('--uncased/--cased', default=True)
@click.option('--pool-size', default=multiprocessing.cpu_count())
def build_corpus_data(dump_db_file, mention_db_file, word_vocab_file, uncased, **kwargs):
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
    entity_linker = EntityLinker(mention_db)

    WikiCorpus.build_corpus_data(dump_db, tokenizer, sentence_tokenizer, entity_linker, **kwargs)


def train_options(func):
    @click.argument('corpus_data_file', type=click.Path())
    @click.argument('entity_vocab_file', type=click.Path(exists=True))
    @click.option('--run-name', type=click.Path(),
                  default=datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    @click.option('--output-dir', type=click.Path(), default='out')
    @click.option('--log-dir', type=click.Path(), default='log')
    @click.option('--mmap', is_flag=True)
    @click.option('--batch-size', default=256)  # BERT default=256
    @click.option('--learning-rate', default=1e-4)  # BERT original=1e-4, recommended for fine-tuning: 2e-5
    @click.option('--warmup-steps', default=10000)
    @click.option('--gradient-accumulation-steps', default=1)
    @click.option('--max-seq-length', default=512)  # BERT default=512
    @click.option('--max-entity-length', default=128)
    @click.option('--short-seq-prob', default=0.1)
    @click.option('--masked-lm-prob', default=0.15)
    @click.option('--max-predictions-per-seq', default=77)  # 512 * 0.15
    @click.option('--num-train-steps', default=300000)
    @click.option('--num-page-split', default=100)
    @click.option('--entity-emb-size', default=768)
    @click.option('--link-prob-bin-size', default=20)
    @click.option('--prior-prob-bin-size', default=20)
    @click.option('--mask-title-words/--no-mask-title-words', default=True)
    @click.option('--bert-model-name', default='bert-base-uncased')
    @click.option('--entity-emb-file', type=click.Path(exists=True), default=None)
    @click.option('--model-type', default='model')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command(name='train')
@train_options
@click.option('-j', '--json-data', default=None)
def train_(entity_vocab_file, json_data, **kwargs):
    from utils.vocab import EntityVocab
    import train

    entity_vocab = EntityVocab(entity_vocab_file)
    if json_data is not None:
        kwargs.update(json.loads(json_data))

    train.train(entity_vocab=entity_vocab, **kwargs)


@cli.command()
@click.argument('data_file', type=click.Path())
@click.option('--batch-size', default=None, type=int)
@click.option('--gradient-accumulation-steps', default=None, type=int)
def resume_training(data_file, batch_size, gradient_accumulation_steps):
    import train

    args = joblib.load(data_file)
    args['model_file'] = data_file.replace('.pkl', '.bin').replace('data', 'model')
    args['optimizer_file'] = data_file.replace('.pkl', '.bin').replace('data', 'optimizer')

    if batch_size is not None:
        args['batch_size'] = batch_size
    if gradient_accumulation_steps is not None:
        args['gradient_accumulation_steps'] = gradient_accumulation_steps

    train.train(**args)


def task_common_options(func):
    @click.argument('model_file', type=click.Path(exists=True))
    @click.argument('word_vocab_file', type=click.Path(exists=True))
    @click.argument('mention_db_file', type=click.Path(exists=True))
    @click.option('--output-dir', type=click.Path())
    @click.option('--data-dir', type=click.Path(exists=True), default='data')
    @click.option('--uncased/--cased', default=True)
    @click.option('--max-seq-length', default=512)
    @click.option('--max-entity-length', default=128)
    @click.option('--min-prior-prob', default=[0.1, 0.3, 0.9], type=float, multiple=True)
    @click.option('--batch-size', default=[32], type=int, multiple=True)
    @click.option('--learning-rate', default=[2e-5, 3e-5, 5e-5], type=float, multiple=True)
    @click.option('--iteration', default=[4.0], type=float, multiple=True)
    @click.option('--eval-batch-size', default=8)
    @click.option('--warmup-proportion', default=0.1)
    @click.option('--lr-decay', is_flag=True)
    @click.option('--seed', default=42)
    @click.option('--gradient-accumulation-steps', default=1)
    @click.option('--fix-entity-emb/--update-entity-emb', default=[True, False], multiple=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def glue_options(func):
    @task_common_options
    @click.option('-t', '--task-name', default='mrpc', type=click.Choice(['cola', 'mnli', 'qnli',
        'mrpc', 'rte', 'qqp', 'scitail', 'sts-b']))
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
