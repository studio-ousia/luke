import click
from pathlib import Path


@click.command()
@click.argument("entity-vocab-path", type=click.Path(exists=True))
@click.argument("output-directory", type=click.Path(exists=True))
def split_entity_vocab(entity_vocab_path: str, output_directory: str):
    with open(entity_vocab_path, "r") as f:
        lines = f.read().strip().split("\n")

    filename_stem = Path(entity_vocab_path).stem
    suffix = Path(entity_vocab_path).suffix
    for start, end in [
        [0, 100],
        [100, 1000],
        [1000, 10000],
        [10000, 100000],
        [100000, 200000],
        [200000, 300000],
        [300000, 400000],
        [400000, 500000],
    ]:
        output_file_path = Path(output_directory) / f"{filename_stem}-{start}_{end}{suffix}"

        with open(output_file_path, "w") as f:
            f.write("\n".join(lines[start:end]))
