import click

input_path = "data/ner_conll/de/deu.dev"


@click.command()
@click.argument("input_path")
@click.option("--encoding", default="ISO-8859-1")
def convert_io_to_bio_format(input_path: str, encoding: str):

    label_history = []
    with open(input_path, "r", encoding=encoding) as in_f, open(input_path + ".bio", "w") as out_f:
        for original_line in in_f:
            line = original_line.strip()
            if not line:
                label_history = []
            else:
                _, _, _, _, label = line.split(" ")
                if label == "O":
                    label_type = "O"
                else:
                    label_type, ent_type = label.split("-")
                    if len(label_history) == 0 or label_history[-1] == "O":
                        converted_label = f"B-{ent_type}"
                        original_line = original_line.replace(label, converted_label)

                label_history.append(label_type)
            out_f.write(original_line)


if __name__ == "__main__":
    convert_io_to_bio_format()
