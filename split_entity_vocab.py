import click
from pathlib import Path


@click.command()
@click.argument("entity-vocab-path", type=click.Path(exists=True))
@click.argument("output-directory", type=click.Path())
def split_entity_vocab(entity_vocab_path: str, output_directory: str):
    print(f"Read vocab from {entity_vocab_path}....")
    with open(entity_vocab_path, "r") as f:
        lines = f.read().strip().split("\n")

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    filename_stem = Path(entity_vocab_path).stem
    suffix = Path(entity_vocab_path).suffix
    for start, end, start_str, end_str in [
        [0, 100, "0", "100"],
        [100, 1000, "100", "1k"],
        [1000, 10000, "1k", "10k"],
        [10000, 100000, "10k", "100k"],
        [100000, 200000, "100k", "200k"],
        [200000, 300000, "200k", "300k"],
        [300000, 400000, "300k", "400k"],
        [400000, 500000, "400k", "500k"],
    ]:
        output_file_path = Path(output_directory) / f"{filename_stem}-{start_str}_{end_str}{suffix}"

        print(f"Write vocab to {output_file_path}....")
        with open(output_file_path, "w") as f:
            f.write("\n".join(lines[start:end]))


if __name__ == "__main__":
    split_entity_vocab()
