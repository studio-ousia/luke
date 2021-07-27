from typing import Set, Dict, List, NamedTuple
import joblib


from transformers.tokenization_utils import PreTrainedTokenizer
from allennlp.data import Token
from allennlp.common import FromParams

from luke.utils.entity_vocab import EntityVocab, Entity
from luke.utils.interwiki_db import InterwikiDB
from .entity_db import EntityDB
from .wiki_link_db import WikiLinkDB


class Mention(NamedTuple):
    entity: Entity
    start: int
    end: int


class WikiMentionDetector(FromParams):
    """
    Detect entity mentions in text from Wikipedia articles.
    """

    def __init__(
        self,
        wiki_link_db_path: str,
        model_redirect_mappings_path: str,
        link_redirect_mappings_path: str,
        entity_vocab_path: str,
        source_language: str = "en",
        inter_wiki_path: str = None,
        multilingual_entity_db_path: Dict[str, str] = None,
        min_mention_link_prob: float = 0.01,
        max_mention_length: int = 10,
    ):
        self.tokenizer = None
        self.wiki_link_db = WikiLinkDB(wiki_link_db_path)
        self.model_redirect_mappings: Dict[str, str] = joblib.load(model_redirect_mappings_path)
        self.link_redirect_mappings: Dict[str, str] = joblib.load(link_redirect_mappings_path)

        self.source_language = source_language
        if inter_wiki_path is not None:
            self.inter_wiki_db = InterwikiDB.load(inter_wiki_path)
        else:
            self.inter_wiki_db = None

        self.entity_vocab = EntityVocab(entity_vocab_path)

        multilingual_entity_db_path = multilingual_entity_db_path or {}
        self.entity_db_dict = {lang: EntityDB(path) for lang, path in multilingual_entity_db_path.items()}

        self.min_mention_link_prob = min_mention_link_prob

        self.max_mention_length = max_mention_length

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def get_mention_candidates(self, title: str) -> Dict[str, str]:
        """
        Returns a dict of [mention, entity (title)]
        """
        title = self.link_redirect_mappings.get(title, title)

        # mention_to_entity
        mention_candidates: Dict[str, str] = {}
        ambiguous_mentions: Set[str] = set()

        for link in self.wiki_link_db.get(title):
            if link.link_prob < self.min_mention_link_prob:
                continue

            link_text = self._normalize_mention(link.text)
            if link_text in mention_candidates and mention_candidates[link_text] != link.title:
                ambiguous_mentions.add(link_text)
                continue

            mention_candidates[link_text] = link.title

        for link_text in ambiguous_mentions:
            del mention_candidates[link_text]
        return mention_candidates

    def _detect_mentions(self, tokens: List[str], mention_candidates: Dict[str, str], language: str) -> List[Mention]:
        mentions = []
        cur = 0
        for start, token in enumerate(tokens):
            if start < cur:
                continue

            for end in range(min(start + self.max_mention_length, len(tokens)), start, -1):

                mention_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
                mention_text = self._normalize_mention(mention_text)
                if mention_text in mention_candidates:
                    cur = end
                    title = mention_candidates[mention_text]
                    title = self.model_redirect_mappings.get(title, title)  # resolve mismatch between two dumps
                    if self.entity_vocab.contains(title, language):
                        mention = Mention(Entity(title, language), start, end)
                        mentions.append(mention)
                    break

        return mentions

    def detect_mentions(self, tokens: List[Token], title: str, language: str) -> List[Mention]:

        if self.tokenizer is None:
            raise RuntimeError("self.tokenizer is None. Did you call self.set_tokenizer()?")

        source_mention_candidates = self.get_mention_candidates(title)

        if language == self.source_language:
            target_mention_candidates = source_mention_candidates
        else:
            if self.inter_wiki_db is None:
                raise ValueError(
                    f"You need InterWikiDB to detect mentions from other languages except for {self.source_language}."
                )
            source_entities = list(source_mention_candidates.values())

            target_entities = []
            for ent in source_entities:
                translated_ent = self.inter_wiki_db.get_title_translation(ent, self.source_language, language)
                if translated_ent is not None:
                    target_entities.append(translated_ent)

            target_mention_candidates = {}
            for target_entity in target_entities:
                for entity, mention, count in self.entity_db_dict[language].query(target_entity):
                    target_mention_candidates[mention] = entity

        target_mentions = self._detect_mentions([t.text for t in tokens], target_mention_candidates, language)

        return target_mentions

    @staticmethod
    def _normalize_mention(text: str):
        return " ".join(text.lower().split(" ")).strip()

    def mentions_to_entity_features(self, tokens: List[Token], mentions: List[Mention]) -> Dict:

        if len(mentions) == 0:
            entity_ids = [0]
            entity_type_ids = [0]
            entity_attention_mask = [0]
            entity_position_ids = [[-1 for y in range(self.max_mention_length)]]
        else:
            entity_ids = [0] * len(mentions)
            entity_type_ids = [0] * len(mentions)
            entity_attention_mask = [1] * len(mentions)
            entity_position_ids = [[-1 for y in range(self.max_mention_length)] for x in range(len(mentions))]

            for i, (entity, start, end) in enumerate(mentions):
                entity_ids[i] = self.entity_vocab.get_id(entity.title, entity.language)
                entity_position_ids[i][: end - start] = range(start, end)

                if tokens[start].type_id is not None:
                    entity_type_ids[i] = tokens[start].type_id

        return {
            "entity_ids": entity_ids,
            "entity_attention_mask": entity_attention_mask,
            "entity_position_ids": entity_position_ids,
            "entity_type_ids": entity_type_ids,
        }
