import functools
import logging
import multiprocessing
import click
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
@click.option('--strip-accents/--no-strip-accents', default=True)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', type=int, default=100)
def build_dump_db(dump_file, out_file, strip_accents, **kwargs):
    from luke.utils import clean_text

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
    from luke.utils.entity_linker import MentionDB

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
@click.option('--min-sentence-len', default=5)
@click.option('--pool-size', default=multiprocessing.cpu_count())
def build_wiki_corpus(dump_db_file, mention_db_file, word_vocab_file, uncased, min_prior_prob,
                      **kwargs):
    from luke.utils.vocab import WordPieceVocab
    from luke.utils.word_tokenizer import WordPieceTokenizer
    from luke.utils.sentence_tokenizer import OpenNLPSentenceTokenizer
    from luke.utils.entity_linker import MentionDB, EntityLinker
    from luke.wiki_corpus import WikiCorpus

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
@click.option('-w', '--white-list', type=click.File(), multiple=True)
@click.option('--white-list-only', is_flag=True)
def build_entity_vocab(corpus_data_file, white_list, **kwargs):
    from luke.wiki_corpus import WikiCorpus
    from luke.utils.vocab import EntityVocab

    corpus = WikiCorpus(corpus_data_file)
    white_list = [line.rstrip() for f in white_list for line in f]
    EntityVocab.build_vocab(corpus, white_list=white_list, **kwargs)


@cli.command()
@click.argument('corpus_data_file', type=click.Path())
@click.argument('entity_vocab_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--bert-model-name', default='bert-base-uncased')
@click.option('-t', '--target-entity-annotation', type=click.Choice(['link', 'mention']), default='link')
@click.option('--single-sentence/--sentence-pair', is_flag=True)
@click.option('--entity-emb-size', default=768)
@click.option('--max-seq-length', default=512)  # BERT default=512
@click.option('--max-entity-length', default=128)
@click.option('--max-mention-length', default=100)
@click.option('--short-seq-prob', default=0.1)
@click.option('--masked-lm-prob', default=0.15)
@click.option('--max-predictions-per-seq', default=77)  # 512 * 0.15
@click.option('--masked-entity-prob', default=0.15)
@click.option('--max-entity-predictions-per-seq', default=19)  # 256 * 0.15
@click.option('--batch-size', default=256)  # BERT default=256
@click.option('--gradient-accumulation-steps', default=1)
@click.option('--learning-rate', default=1e-4)  # BERT original=1e-4, recommended for fine-tuning: 2e-5
@click.option('--lr-decay/--no-lr-decay', default=False)
@click.option('--warmup-steps', default=0)
@click.option('--fix-bert-weights', is_flag=True)
@click.option('--allocate-gpu-for-optimizer', is_flag=True)
@click.option('--num-train-steps', default=300000)
@click.option('--num-page-chunks', default=100)
@click.option('--log-dir', type=click.Path(), default=None)
@click.option('--model-file', type=click.Path(exists=True), default=None)
@click.option('--optimizer-file', type=click.Path(exists=True), default=None)
def run_training(**kwargs):
    from luke import train

    train.run_training(**kwargs)
