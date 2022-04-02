from dataclasses import dataclass
from typing import List

from wikipedia2vec.dump_db import DumpDB


@dataclass
class WikiLink:
    text: str
    title: str
    start: int
    end: int


@dataclass
class Paragraph:
    text: str
    wiki_links: List[WikiLink] = None
    abstract: str = None


SAMPLE_PARAGRAPHS = {
    "Japan": [
        Paragraph(
            "Japan is an island country in East Asia. It is situated in the northwest Pacific Ocean.",
            wiki_links=[
                WikiLink("Japan", "Japan", 0, 5),
                WikiLink("East Asia", "East Asia", 30, 39),
                WikiLink("Pacific Ocean", "Pacific Ocean", 73, 86),
            ],
        )
    ],
    "Studio Ousia": [
        Paragraph(
            "Studio Ousia develops advanced multilingual natural language AI.",
            wiki_links=[
                WikiLink("Studio Ousia", "Studio Ousia", 0, 12),
                WikiLink("AI", "Artificial Intelligence", 61, 63),
            ],
        ),
        Paragraph(
            "Our award-winning AI will accelerate your business.",
            wiki_links=[
                WikiLink("AI", "Artificial Intelligence", 18, 20),
            ],
        ),
    ],
}


class DummyDumpDB(DumpDB):

    language = None

    def __init__(self):
        pass

    def get_paragraphs(self, page_title: str):
        return SAMPLE_PARAGRAPHS[page_title]

    def is_disambiguation(self, title: str):
        return False

    def is_redirect(self, title: str):
        return False

    def resolve_redirect(self, title: str):
        return title

    def titles(self):
        return list(SAMPLE_PARAGRAPHS.keys())
