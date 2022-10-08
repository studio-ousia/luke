import click
from wikipedia2vec.dump_db import DumpDB

from dataset import load_dataset


@click.command()
@click.option("--db-file", type=click.Path(exists=True), required=True)
@click.option("--dataset-dir", type=click.Path(exists=True), required=True)
@click.option("--output-file", type=click.File("w"), required=True)
@click.option("--max-candidate-length", type=int, default=30)
def create_candidate_data(db_file, dataset_dir, output_file, max_candidate_length):
    dump_db = DumpDB(db_file)

    titles = set()
    valid_titles = frozenset(dump_db.titles())

    dataset = load_dataset(dataset_dir)
    for documents in dataset.get_all_datasets():
        for document in documents:
            for mention in document.mentions:
                candidates = mention.candidates[:max_candidate_length]
                for candidate in candidates:
                    title = dump_db.resolve_redirect(candidate.title)
                    if title in valid_titles:
                        titles.add(title)

    for title in titles:
        output_file.write(title + "\n")


if __name__ == "__main__":
    create_candidate_data()
