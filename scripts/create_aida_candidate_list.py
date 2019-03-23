# -*- coding: utf-8 -*-

import click
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB

from utils.aida_conll import AIDACoNLLReader
from utils.entity_linker import MentionDB
from utils import clean_text


@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('mention_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File('w'))
@click.option('--max-candidate-size', default=30)
def main(dataset_dir, dump_db_file, mention_db_file, out_file, max_candidate_size):
    dump_db = DumpDB(dump_db_file)
    mention_db = MentionDB.load(mention_db_file)

    titles = set()
    valid_titles = frozenset(dump_db.titles())

    reader = AIDACoNLLReader(dataset_dir)
    for document in tqdm(reader.get_documents()):
        for mention in document.mentions:
            try:
                db_candidates = mention_db.query(clean_text(mention.text))
            except KeyError:
                db_candidates = []

            prior_probs = {c.title: c.prior_prob for c in db_candidates}
            candidates = [dump_db.resolve_redirect(title) for title in mention.candidates]
            sorted_candidates = sorted(candidates, key=lambda title: prior_probs.get(title, 0.0),
                                       reverse=True)

            for candidate in sorted_candidates[:max_candidate_size]:
                if candidate in valid_titles:
                    titles.add(candidate)

    for title in titles:
        out_file.write(title + '\n')


if __name__ == '__main__':
    main()
