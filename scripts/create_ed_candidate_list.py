# -*- coding: utf-8 -*-

import click
from wikipedia2vec.dump_db import DumpDB

from utils.entity_disambiguation.dataset import EntityDisambiguationDataset


@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File('w'))
@click.option('--max-candidate-size', default=30)
def main(dataset_dir, dump_db_file, out_file, max_candidate_size):
    dump_db = DumpDB(dump_db_file)

    titles = set()
    valid_titles = frozenset(dump_db.titles())

    reader = EntityDisambiguationDataset(dataset_dir)
    for documents in reader.get_all_datasets():
        for document in documents:
            for mention in document.mentions:
                for candidate in  sorted(mention.candidates, key=lambda c: c.prior_prob,
                    reverse=True)[:max_candidate_size]:
                    title = dump_db.resolve_redirect(candidate.title)
                    if title in valid_titles:
                        titles.add(title)

    for title in titles:
        out_file.write(title + '\n')


if __name__ == '__main__':
    main()
