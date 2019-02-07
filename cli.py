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

import train
from utils import clean_text
from utils.entity_linker import MentionDB
from utils.sentence_tokenizer import OpenNLPSentenceTokenizer
from utils.word_tokenizer import WordPieceTokenizer
from utils.vocab import WordPieceVocab, EntityVocab
from wiki_corpus import WikiCorpus

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
    dump_reader = WikiDumpReader(dump_file)
    func = clean_text
    if strip_accents:
        func = functools.partial(clean_text, strip_accents=True)

    DumpDB.build(dump_reader, out_file, preprocess_func=func, **kwargs)


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--max-mention-len', default=100)
@click.option('--min-link-prob', default=0.05)
@click.option('--min-prior-prob', default=0.05)
@click.option('--min-link-count', default=0)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=30)
def build_mention_db(dump_db_file, out_file, **kwargs):
    dump_db = DumpDB(dump_db_file)
    mention_db = MentionDB.build(dump_db, **kwargs)
    mention_db.save(out_file)


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--vocab-size', default=1000000)
@click.option('-i', '--include', type=click.File(), multiple=True)
def build_entity_vocab(dump_db_file, include, **kwargs):
    dump_db = DumpDB(dump_db_file)
    white_list = [line.rstrip() for f in include for line in f]
    EntityVocab.build_vocab(dump_db, white_list=white_list, **kwargs)


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('mention_db_file', type=click.Path(exists=True))
@click.argument('word_vocab_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--min-sentence-len', default=10)
@click.option('--min-link-prob', default=0.1)
@click.option('--min-prior-prob', default=0.1)
@click.option('--uncased/--cased', default=True)
@click.option('--pool-size', default=multiprocessing.cpu_count())
def build_corpus_data(dump_db_file, mention_db_file, word_vocab_file, uncased, **kwargs):
    dump_db = DumpDB(dump_db_file)
    mention_db = MentionDB.load(mention_db_file)
    word_vocab = WordPieceVocab(word_vocab_file)
    tokenizer = WordPieceTokenizer(word_vocab, lowercase=uncased)
    sentence_tokenizer = OpenNLPSentenceTokenizer()

    WikiCorpus.build_corpus_data(dump_db, mention_db, tokenizer, sentence_tokenizer, **kwargs)


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
    @click.option('--fp16', is_flag=True)
    @click.option('--static-loss-scale', default=128.0, type=float)
    @click.option('--fp16-emb/--fp32-emb', default=True)
    @click.option('--max-seq-length', default=512)  # BERT default=512
    @click.option('--max-entity-length', default=128)
    @click.option('--short-seq-prob', default=0.1)
    @click.option('--masked-lm-prob', default=0.15)
    @click.option('--max-predictions-per-seq', default=77)  # 512 * 0.15
    @click.option('--num-train-steps', default=300000)
    @click.option('--num-page-split', default=100)
    @click.option('--entity-emb-size', default=768)
    @click.option('--bert-model-name', default='bert-base-uncased')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command(name='train')
@train_options
@click.option('-j', '--json-data', default=None)
def train_model(entity_vocab_file, json_data, **kwargs):
    entity_vocab = EntityVocab(entity_vocab_file)
    if json_data is not None:
        kwargs.update(json.loads(json_data))
    if kwargs['fp16']:
        kwargs['fp16_emb'] = False

    train.train(entity_vocab=entity_vocab, **kwargs)


@cli.command()
@click.argument('data_file', type=click.Path())
@click.argument('model_file', type=click.Path())
@click.argument('optimizer_file', type=click.Path())
@click.option('--batch-size', default=None, type=int)
@click.option('--gradient-accumulation-steps', default=None, type=int)
def resume_training(data_file, model_file, optimizer_file, batch_size, gradient_accumulation_steps):
    args = joblib.load(data_file)

    if 'model_state_dict_file' in args:
        del args['model_state_dict_file']
    if 'optimizer_state_dict_file' in args:
        del args['optimizer_state_dict_file']

    args['model_file'] = model_file
    args['optimizer_file'] = optimizer_file

    if batch_size is not None:
        args['batch_size'] = batch_size
    if gradient_accumulation_steps is not None:
        args['gradient_accumulation_steps'] = gradient_accumulation_steps

    train.train(**args)


if __name__ == '__main__':
    cli()
