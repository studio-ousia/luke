import click
from wikipedia2vec.dump_db import DumpDB


@click.command()
@click.option("--db-file", type=click.Path(exists=True), required=True)
@click.option("--output-file", type=click.File(mode="w"), required=True)
def create_title_data(db_file, output_file):
    dump_db = DumpDB(db_file)

    for title in dump_db.titles():
        output_file.write(f"{title}\n")


if __name__ == "__main__":
    create_title_data()
