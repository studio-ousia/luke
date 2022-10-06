import click
from wikipedia2vec.dump_db import DumpDB


@click.command()
@click.option("--db-file", type=click.Path(exists=True), required=True)
@click.option("--output-file", type=click.File(mode="w"))
def create_redirect_data(db_file, output_file):
    dump_db = DumpDB(db_file)

    for src, dest in dump_db.redirects():
        output_file.write(f"{src}\t{dest}\n")


if __name__ == "__main__":
    create_redirect_data()
