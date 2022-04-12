import json

import click

from examples.utils.wiki_entity_linker.mention_candidate_translator import MentionCandidatesTranslator


@click.command()
@click.argument("data-path", type=str)
@click.argument("source-language", type=str)
@click.argument("target-language", type=str)
@click.argument("save-path", type=str)
@click.option("--inter_wiki_path", type=str, default=None)
@click.option("--target_entity_db_path", type=str, default=None)
def translate_mention_candidates(
    data_path: str,
    source_language: str,
    target_language: str,
    save_path: str,
    inter_wiki_path: str,
    target_entity_db_path: str,
):

    mention_candidate_translator = MentionCandidatesTranslator(
        inter_wiki_path=inter_wiki_path, multilingual_entity_db_path={target_language: target_entity_db_path}
    )

    source_mention_candidates = json.load(open(data_path))
    target_mention_candidates = dict()

    for title, mention_candidates in source_mention_candidates.items():
        target_mention_candidates[title] = mention_candidate_translator(
            mention_candidates, source_language=source_language, target_language=target_language
        )

    with open(save_path, "w") as f:
        json.dump(target_mention_candidates, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    translate_mention_candidates()
