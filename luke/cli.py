import functools
import logging
import multiprocessing
import click
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader

logger = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbose', is_flag=True)
def cli(verbose):
    fmt = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)


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
@click.argument('out_file', type=click.Path())
@click.option('--max-mention-length', default=20)
def build_entity_linker_from_p_e_m_file(p_e_m_file, dump_db_file, **kwargs):
    from luke.utils.entity_linker import EntityLinker, BertLowercaseNormalizer

    dump_db = DumpDB(dump_db_file)
    tokenizer = BasicTokenizer(do_lower_case=False)
    normalizer = BertLowercaseNormalizer()
    EntityLinker.build_from_p_e_m_file(p_e_m_file, dump_db, tokenizer, normalizer, **kwargs)


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
    @click.option('--bert-model-name', default='bert-base-uncased')
    @click.option('--single-sentence', is_flag=True)
    @click.option('--max-seq-length', default=512)
    @click.option('--max-entity-length', default=128)
    @click.option('--max-mention-length', default=30)
    @click.option('--short-seq-prob', default=0.1)
    @click.option('--masked-lm-prob', default=0.15)
    @click.option('--masked-entity-prob', default=0.3)
    @click.option('--batch-size', default=256)
    @click.option('--gradient-accumulation-steps', default=1)
    @click.option('--learning-rate', default=1e-4)
    @click.option('--lr-schedule', type=click.Choice(['none', 'warmup_cosine', 'warmup_constant', 'warmup_linear']),
                  default='warmup_constant')
    @click.option('--warmup-steps', default=0)
    @click.option('--fix-bert-weights', is_flag=True)
    @click.option('--optimizer-on-cpu', is_flag=True)
    @click.option('--num-train-steps', default=300000)
    @click.option('--num-page-chunks', default=100)
    @click.option('--log-dir', type=click.Path(), default=None)
    @click.option('--model-file', type=click.Path(exists=True), default=None)
    @click.option('--optimizer-file', type=click.Path(exists=True), default=None)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@common_train_options
def run_training(**kwargs):
    from luke import train
    train.run_training(**kwargs)


@cli.command()
@click.option('--max-candidate-length', default=10)
@click.option('--min-candidate-prior-prob', default=0.01)
@click.option('--num-el-hidden-layers', default=3)
@click.option('--entity-selector-softmax-temp', default=0.1)
@common_train_options
def run_e2e_training(**kwargs):
    from luke import train
    train.run_e2e_training(**kwargs)
