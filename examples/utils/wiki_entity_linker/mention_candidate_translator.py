from typing import Dict

from allennlp.common import FromParams

from luke.utils.interwiki_db import InterwikiDB
from .entity_db import EntityDB


class MentionCandidatesTranslator(FromParams):
    def __init__(
        self,
        inter_wiki_path: str,
        multilingual_entity_db_path: Dict[str, str] = None,
    ):
        self.inter_wiki_db = InterwikiDB.load(inter_wiki_path)

        multilingual_entity_db_path = multilingual_entity_db_path or {}
        self.entity_db_dict = {lang: EntityDB(path) for lang, path in multilingual_entity_db_path.items()}

    def __call__(
        self, mention_candidates: Dict[str, str], source_language: str, target_language: str
    ) -> Dict[str, str]:

        assert target_language in self.entity_db_dict
        source_titles = list(mention_candidates.values())

        target_titles = []
        for title in source_titles:
            translated_title = self.inter_wiki_db.get_title_translation(title, source_language, target_language)
            if translated_title is not None:
                target_titles.append(translated_title)

        target_mention_candidates = dict()
        ambiguous_mentions = set()
        entity_db = self.entity_db_dict[target_language]
        for target_title in target_titles:
            for _, mention, count in entity_db.query(target_title):
                if mention in target_mention_candidates:
                    ambiguous_mentions.add(mention)
                    del target_mention_candidates[mention]

                if mention not in ambiguous_mentions:
                    target_mention_candidates[mention] = target_title

        return target_mention_candidates
