import json

import click
import tqdm

from examples.utils.wiki_entity_linker.wiki_entity_linker import MentionCandidatesGenerator


def get_titles(data_path: str):

    data = json.load(open(data_path))
    for instance in data["data"]:
        title = instance["title"]
        title = title.replace("_", " ")
        yield title


@click.command()
@click.argument("data-path", type=str)
@click.argument("save-path", type=str)
@click.option("--wiki_link_db_path", type=str, default=None)
@click.option("--link_redirect_mappings_path", type=str, default=None)
@click.option("--model_redirect_mappings_path", type=str, default=None)
def make_mention_candidates(
    data_path: str,
    save_path: str,
    wiki_link_db_path: str,
    link_redirect_mappings_path: str,
    model_redirect_mappings_path: str,
):

    mention_candidate_generator = MentionCandidatesGenerator(
        wiki_link_db_path=wiki_link_db_path,
        link_redirect_mappings_path=link_redirect_mappings_path,
        model_redirect_mappings_path=model_redirect_mappings_path,
    )

    mention_candidates = dict()
    for title in tqdm.tqdm(get_titles(data_path)):
        mention_candidates[title] = mention_candidate_generator.get_mention_candidates(title)

    with open(save_path, "w") as f:
        json.dump(mention_candidates, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    make_mention_candidates()
