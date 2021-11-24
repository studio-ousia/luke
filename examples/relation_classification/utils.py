import json
import os

from tqdm import tqdm
from transformers import RobertaTokenizer

HEAD_TOKEN = "[HEAD]"
TAIL_TOKEN = "[TAIL]"


class InputExample(object):
    def __init__(self, id_, text, span_a, span_b, type_a, type_b, label):
        self.id = id_
        self.text = text
        self.span_a = span_a
        self.span_b = span_b
        self.type_a = type_a
        self.type_b = type_b
        self.label = label


class InputFeatures(object):
    def __init__(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        label,
    ):
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.label = label


class DatasetProcessor(object):
    def get_train_examples(self, data_dir):
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(data_dir, "test")

    def get_label_list(self, data_dir):
        labels = set()
        for example in self.get_train_examples(data_dir):
            labels.add(example.label)
        labels.discard("no_relation")
        return ["no_relation"] + sorted(labels)

    def _create_examples(self, data_dir, set_type):
        with open(os.path.join(data_dir, set_type + ".json"), "r") as f:
            data = json.load(f)

        examples = []
        for i, item in enumerate(data):
            tokens = item["token"]
            token_spans = dict(
                subj=(item["subj_start"], item["subj_end"] + 1), obj=(item["obj_start"], item["obj_end"] + 1)
            )

            if token_spans["subj"][0] < token_spans["obj"][0]:
                entity_order = ("subj", "obj")
            else:
                entity_order = ("obj", "subj")

            text = ""
            cur = 0
            char_spans = dict(subj=[None, None], obj=[None, None])
            for target_entity in entity_order:
                token_span = token_spans[target_entity]
                text += " ".join(tokens[cur : token_span[0]])
                if text:
                    text += " "
                char_spans[target_entity][0] = len(text)
                text += " ".join(tokens[token_span[0] : token_span[1]]) + " "
                char_spans[target_entity][1] = len(text)
                cur = token_span[1]
            text += " ".join(tokens[cur:])
            text = text.rstrip()

            examples.append(
                InputExample(
                    "%s-%s" % (set_type, i),
                    text,
                    char_spans["subj"],
                    char_spans["obj"],
                    item["subj_type"],
                    item["obj_type"],
                    item["relation"],
                )
            )

        return examples


def convert_examples_to_features(examples, label_list, tokenizer, max_mention_length):
    label_map = {l: i for i, l in enumerate(label_list)}

    def tokenize(text):
        text = text.rstrip()
        if isinstance(tokenizer, RobertaTokenizer):
            return tokenizer.tokenize(text, add_prefix_space=True)
        else:
            return tokenizer.tokenize(text)

    features = []
    for example in tqdm(examples):
        if example.span_a[1] < example.span_b[1]:
            span_order = ("span_a", "span_b")
        else:
            span_order = ("span_b", "span_a")

        tokens = [tokenizer.cls_token]
        cur = 0
        token_spans = {}
        for span_name in span_order:
            span = getattr(example, span_name)
            tokens += tokenize(example.text[cur : span[0]])
            start = len(tokens)
            tokens.append(HEAD_TOKEN if span_name == "span_a" else TAIL_TOKEN)
            tokens += tokenize(example.text[span[0] : span[1]])
            tokens.append(HEAD_TOKEN if span_name == "span_a" else TAIL_TOKEN)
            token_spans[span_name] = (start, len(tokens))
            cur = span[1]

        tokens += tokenize(example.text[cur:])
        tokens.append(tokenizer.sep_token)

        word_ids = tokenizer.convert_tokens_to_ids(tokens)
        word_attention_mask = [1] * len(tokens)
        word_segment_ids = [0] * len(tokens)

        entity_ids = [1, 2]
        entity_position_ids = []
        for span_name in ("span_a", "span_b"):
            span = token_spans[span_name]
            position_ids = list(range(span[0], span[1]))[:max_mention_length]
            position_ids += [-1] * (max_mention_length - span[1] + span[0])
            entity_position_ids.append(position_ids)

        entity_segment_ids = [0, 0]
        entity_attention_mask = [1, 1]

        features.append(
            InputFeatures(
                word_ids=word_ids,
                word_segment_ids=word_segment_ids,
                word_attention_mask=word_attention_mask,
                entity_ids=entity_ids,
                entity_position_ids=entity_position_ids,
                entity_segment_ids=entity_segment_ids,
                entity_attention_mask=entity_attention_mask,
                label=label_map[example.label],
            )
        )

    return features
