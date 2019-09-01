import click
from wikipedia2vec.dump_db import DumpDB

from dataset import EntityDisambiguationDataset

@click.group()
def cli():
    pass


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File('w'))
@click.option('--data-dir', type=click.Path(exists=True), default='data/entity-disambiguation')
def create_candidate_list(dump_db_file, out_file, data_dir):
    dump_db = DumpDB(dump_db_file)

    titles = set()
    valid_titles = frozenset(dump_db.titles())

    reader = EntityDisambiguationDataset(data_dir)
    for documents in reader.get_all_datasets():
        for document in documents:
            for mention in document.mentions:
                candidates = mention.candidates
                for candidate in candidates:
                    title = dump_db.resolve_redirect(candidate.title)
                    if title in valid_titles:
                        titles.add(title)

    for title in titles:
        out_file.write(title + '\n')


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File(mode='w'))
def create_title_list(dump_db_file, out_file):
    dump_db = DumpDB(dump_db_file)

    for title in dump_db.titles():
        out_file.write(f'{title}\n')


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File(mode='w'))
def create_redirect_tsv(dump_db_file, out_file):
    dump_db = DumpDB(dump_db_file)

    for (src, dest) in dump_db.redirects():
        out_file.write(f'{src}\t{dest}\n')


if __name__ == '__main__':
    cli()
