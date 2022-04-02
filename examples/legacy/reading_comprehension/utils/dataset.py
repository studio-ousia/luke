"""
The original version of this code is based on the following:
https://github.com/huggingface/transformers/blob/23c6998bf46e43092fc59543ea7795074a720f08/src/transformers/data/processors/squad.py#L38
"""
import json
import logging
import os
import urllib.parse

logger = logging.getLogger(__name__)


class SquadExample(object):
    def __init__(self, qas_id, title, question_text, context_text, answers, is_impossible=False):
        self.qas_id = qas_id
        self.title = title.replace("_", " ")
        self.question_text = question_text
        self.context_text = context_text
        self.answers = answers
        self.is_impossible = is_impossible

        self.start_positions = []
        self.end_positions = []
        self.answer_texts = []

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        for c in self.context_text:
            if self._is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        self.doc_tokens = doc_tokens

        for answer in answers:
            self.start_positions.append(char_to_word_offset[answer["answer_start"]])
            self.end_positions.append(
                char_to_word_offset[min(answer["answer_start"] + len(answer["text"]) - 1, len(char_to_word_offset) - 1)]
            )
            self.answer_texts.append(answer["text"])

    @staticmethod
    def _is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False


class SquadProcessor(object):
    train_file = None
    dev_file = None

    def get_train_examples(self, data_dir):
        with open(os.path.join(data_dir, self.train_file)) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data)

    def get_dev_examples(self, data_dir, filename=None):
        with open(os.path.join(data_dir, self.dev_file if filename is None else filename)) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data)

    # def __init__(self, qas_id, title, question_text, context_text, answers, is_impossible=False):
    def _create_examples(self, input_data):
        return [
            SquadExample(
                qas_id=qa["id"],
                title=urllib.parse.unquote(entry["title"].replace("_", " ")),
                question_text=qa["question"],
                context_text=para["context"],
                answers=qa.get("answers", []),
                is_impossible=qa.get("is_impossible", False),
            )
            for entry in input_data
            for para in entry["paragraphs"]
            for qa in para["qas"]
        ]


class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"
