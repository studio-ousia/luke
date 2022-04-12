from abc import abstractmethod, ABCMeta
from typing import Dict, List, Tuple, NamedTuple
import json

from allennlp.common import Registrable
from allennlp.data import Token
from transformers.tokenization_utils import PreTrainedTokenizer

from .mention_candidate_generator import MentionCandidatesGenerator, normalize_mention
from .mention_candidate_translator import MentionCandidatesTranslator
from luke.utils.entity_vocab import PAD_TOKEN, Entity, EntityVocab


class Mention(NamedTuple):
    entity: Entity
    start: int
    end: int


class WikiEntityLinker(Registrable, metaclass=ABCMeta):
    def __init__(
        self,
        entity_vocab_path: str,
        max_mention_length: int = 10,
    ):
        self.entity_vocab = EntityVocab(entity_vocab_path)
        self.max_mention_length = max_mention_length
        self.tokenizer = None

    def link_entities(self, tokens: List[Token], token_language: str, title: str, title_language: str) -> List[Mention]:
        mention_candidates = self.get_mention_candidates(title, title_language, token_language)
        return self._link_entities([t.text for t in tokens], mention_candidates, language=token_language)

    @abstractmethod
    def get_mention_candidates(self, title: str, title_language: str, token_language: str) -> Dict[str, str]:
        raise NotImplementedError()

    def _link_entities(self, tokens: List[str], mention_candidates: Dict[str, str], language: str) -> List[Mention]:
        mentions = []
        cur = 0
        for start, token in enumerate(tokens):
            if start < cur:
                continue

            for end in range(min(start + self.max_mention_length, len(tokens)), start, -1):

                mention_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
                mention_text = normalize_mention(mention_text)

                title = mention_candidates.get(mention_text, None)
                if title is None:
                    continue

                cur = end
                if self.entity_vocab.contains(title, language):
                    mention = Mention(Entity(title, language), start, end)
                    mentions.append(mention)
                break

        return mentions

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def mentions_to_entity_features(self, tokens: List[Token], mentions: List[Mention]) -> Dict:

        if len(mentions) == 0:
            entity_ids = [self.entity_vocab.special_token_ids[PAD_TOKEN]]
            entity_segment_ids = [0]
            entity_attention_mask = [0]
            entity_position_ids = [[-1 for _ in range(self.max_mention_length)]]
        else:
            entity_ids = [0] * len(mentions)
            entity_segment_ids = [0] * len(mentions)
            entity_attention_mask = [1] * len(mentions)
            entity_position_ids = [[-1 for _ in range(self.max_mention_length)] for x in range(len(mentions))]

            for i, (entity, start, end) in enumerate(mentions):
                entity_ids[i] = self.entity_vocab.get_id(entity.title, entity.language)
                entity_position_ids[i][: end - start] = range(start, end)

                if tokens[start].type_id is not None:
                    entity_segment_ids[i] = tokens[start].type_id

        return {
            "entity_ids": entity_ids,
            "entity_attention_mask": entity_attention_mask,
            "entity_position_ids": entity_position_ids,
            "entity_segment_ids": entity_segment_ids,
        }


@WikiEntityLinker.register("json")
class JsonWikiEntityLinker(WikiEntityLinker):
    def __init__(
        self,
        mention_candidate_json_file_paths: Dict[Tuple[str, str], str],
        entity_vocab_path: str,
        max_mention_length: int = 10,
    ):
        super().__init__(entity_vocab_path=entity_vocab_path, max_mention_length=max_mention_length)
        self.mention_candidates = {
            title_token_language: json.load(open(path))
            for (title_token_language), path in mention_candidate_json_file_paths.items()
        }

    def get_mention_candidates(self, title: str, title_language: str, token_language: str) -> Dict[str, str]:
        mention_candidates = self.mention_candidates[f"{title_language}-{token_language}"][title]
        return mention_candidates


@WikiEntityLinker.register("mention_generator")
class GenerativeWikiEntityLinker(WikiEntityLinker):
    def __init__(
        self,
        mention_candidate_generators: Dict[str, MentionCandidatesGenerator],
        entity_vocab_path: str,
        mention_candidates_translator: MentionCandidatesTranslator = None,
        max_mention_length: int = 10,
    ):
        super().__init__(entity_vocab_path=entity_vocab_path, max_mention_length=max_mention_length)
        self.mention_candidate_generators = mention_candidate_generators
        self.mention_candidates_translator = mention_candidates_translator

    def get_mention_candidates(self, title: str, title_language: str, token_language: str) -> Dict[str, str]:
        assert title_language in self.mention_candidate_generators

        mention_candidates = self.mention_candidate_generators[title_language].get_mention_candidates(title)

        if token_language != title_language:
            assert self.mention_candidates_translator is not None
            mention_candidates = self.mention_candidates_translator(
                mention_candidates, source_language=title_language, target_language=token_language
            )
        return mention_candidates
