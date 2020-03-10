"""
The original version of this code is based on the following:
https://github.com/huggingface/transformers/blob/23c6998bf46e43092fc59543ea7795074a720f08/src/transformers/data/processors/squad.py#L38
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

RECORD_PLACEHOLDER_TOKEN = '[placeholder]'
RECORD_HIGHLIGHT_TOKEN = '[highlight]'


class BaseExample(object):
    def __init__(self, qas_id, question_text, context_text, answers, is_impossible=False):
        self.qas_id = qas_id
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
            self.start_positions.append(char_to_word_offset[answer['answer_start']])
            self.end_positions.append(char_to_word_offset[min(answer['answer_start'] + len(answer['text']) - 1,
                                                              len(char_to_word_offset) - 1)])
            self.answer_texts.append(answer['text'])

    @staticmethod
    def _is_whitespace(c):
        if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F:
            return True
        return False


class RecordExample(BaseExample):
    def __init__(self, qas_id, question_text, context_text, answers, entities):
        super(RecordExample, self).__init__(qas_id, question_text, context_text, answers)
        self.entities = entities


class SquadExample(BaseExample):
    pass


class RecordProcessor(object):
    train_file = 'train.json'
    dev_file = 'dev.json'

    def __init__(self, placeholder_token=RECORD_PLACEHOLDER_TOKEN, highlight_token=RECORD_HIGHLIGHT_TOKEN):
        self.placeholder_token = placeholder_token
        self.highlight_token = highlight_token

    def get_train_examples(self, data_dir):
        with open(os.path.join(data_dir, self.train_file)) as reader:
            input_data = json.load(reader)['data']
        return self._create_examples(input_data)

    def get_dev_examples(self, data_dir):
        with open(os.path.join(data_dir, self.dev_file)) as reader:
            input_data = json.load(reader)['data']
        return self._create_examples(input_data)

    def _create_examples(self, input_data):
        examples = []
        for entry in input_data:
            context_text = entry['passage']['text']
            entities = entry['passage']['entities']
            for qa in entry['qas']:
                qas_id = qa['id']
                question_text = qa['query']
                answers = qa.get('answers', [])
                for answer in answers:
                    answer['answer_start'] = answer['start']  # for compatibility for SQuAD

                example = RecordExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answers=answers,
                    entities=entities,
                )
                example.question_text = example.question_text.replace('@placeholder', self.placeholder_token)
                example.doc_tokens = [t.replace('@highlight', self.highlight_token) for t in example.doc_tokens]

                examples.append(example)

        return examples


class SquadProcessor(object):
    train_file = None
    dev_file = None

    def get_train_examples(self, data_dir):
        with open(os.path.join(data_dir, self.train_file)) as reader:
            input_data = json.load(reader)['data']
        return self._create_examples(input_data)

    def get_dev_examples(self, data_dir, filename=None):
        with open(os.path.join(data_dir, self.dev_file if filename is None else filename)) as reader:
            input_data = json.load(reader)['data']
        return self._create_examples(input_data)

    def _create_examples(self, input_data):
        return [SquadExample(qa['id'], qa['question'], para['context'], qa.get('answers', []),
                             qa.get('is_impossible', False))
                for entry in input_data for para in entry['paragraphs'] for qa in para['qas']]


class SquadV1Processor(SquadProcessor):
    train_file = 'train-v1.1.json'
    dev_file = 'dev-v1.1.json'


class SquadV2Processor(SquadProcessor):
    train_file = 'train-v2.0.json'
    dev_file = 'dev-v2.0.json'
