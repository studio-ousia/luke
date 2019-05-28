import click
import logging
import random
import numpy as np
import torch
import json
from tqdm import tqdm, trange
import collections
import os
import math

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from luke.utils.vocab import WordPieceVocab, EntityVocab
from luke.utils.word_tokenizer import WordPieceTokenizer, BasicTokenizer
from luke.optimization import BertAdam
from luke.model import LukeConfig
from luke.utils.entity_linker import EntityLinker, MentionDB
from squad_model import LukeForQuestionAnswering

from squad_dataset import read_squad_examples, convert_examples_to_features

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + \
                    result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(
                    pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(
                    orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, case):
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    do_lower_case = not case
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


@click.command()
@click.argument('word_vocab_file', type=click.Path(exists=True))
@click.argument('entity_vocab_file', type=click.Path(exists=True))
@click.argument('mention_db_file', type=click.Path(exists=True))
@click.argument('model_file', type=click.Path())
@click.option('--train-file-path', type=click.Path(exists=True), default='squad/train-v1.1.json')
@click.option('--dev-file-path', type=click.Path(exists=True), default='squad/dev-v1.1.json')
@click.option('--output_dir', default='/tmp/squad')
@click.option('--cased/--uncased', default=False)
@click.option('--max-seq-length', default=512)
@click.option('--max-query-length', default=64)
@click.option('--max-entity-length', default=128)
@click.option('--max-mention-length', default=20)
@click.option('--doc-stride', default=128)
@click.option('--batch-size', default=32)
@click.option('--eval-batch-size', default=8)
@click.option('--learning-rate', default=1e-5)
@click.option('--num-train-epochs', default=3.0)
@click.option('--iteration', default=3.0)
@click.option('--warmup-proportion', default=0.1)
@click.option('--lr-decay/--no-lr-decay', default=True)
@click.option('--seed', default=42)
@click.option('--gradient-accumulation-steps', default=32)
@click.option('--fix-entity-emb/--update-entity-emb', default=True)
@click.option('--use-entities/--no-entities', default=True)
@click.option('--min-prior-prob', default=0.1)
@click.option('--n-best-size', default=20)
@click.option('--max-answer-length', default=30)
@click.option('--version-2-with-negative', default=False)
@click.option('--null_score_diff_threshold', default=0.0)
def run(word_vocab_file, entity_vocab_file, mention_db_file, model_file,
        train_file_path, dev_file_path, cased, output_dir,
        max_seq_length, max_entity_length, max_mention_length, batch_size, eval_batch_size,
        learning_rate, iteration, warmup_proportion, lr_decay, seed, gradient_accumulation_steps,
        fix_entity_emb, use_entities, min_prior_prob, num_train_epochs,
        doc_stride, max_query_length, max_answer_length,
        n_best_size, version_2_with_negative, null_score_diff_threshold):
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=log_format)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    word_vocab = WordPieceVocab(word_vocab_file)
    tokenizer = WordPieceTokenizer(word_vocab, not cased)
    entity_vocab = EntityVocab(entity_vocab_file)
    mention_db = MentionDB.load(mention_db_file)
    entity_linker = EntityLinker(mention_db, min_prior_prob=min_prior_prob)

    json_file = model_file + '.json'
    with open(json_file) as f:
        model_data = json.load(f)

    model_config = model_data['model_config']
    config = LukeConfig(**model_config)
    logger.info('Model configuration: %s', config)

    model = LukeForQuestionAnswering(config)

    state_dict = torch.load(model_file + '.bin', map_location='cpu')
    model_state_dict = model.state_dict()
    model_state_dict.update(
        {k: v for k, v in state_dict.items() if k in model_state_dict})
    model.load_state_dict(model_state_dict)
    # model.load_state_dict(state_dict, strict=False)
    # del state_dict, model_state_dict

    logger.info('Fix entity embeddings during training: %s', fix_entity_emb)
    model.embeddings.word_embeddings.sparse = True
    model.entity_embeddings.entity_embeddings.sparse = True
    if fix_entity_emb:
        model.entity_embeddings.entity_embeddings.weight.requires_grad = False

    device = torch.device("cuda")
    model.to(device)

    train_batch_size = int(batch_size / gradient_accumulation_steps)

    train_examples = read_squad_examples(
        input_file=train_file_path, is_training=True, version_2_with_negative=version_2_with_negative)
    # num_train_optimization_steps = int(
    #     len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
    num_train_steps = int(len(train_examples) / batch_size * iteration)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_parameters, lr=learning_rate, lr_decay=lr_decay,
                         device=device, warmup=warmup_proportion, t_total=num_train_steps)
    global_step = 0

    train_features = convert_examples_to_features(
        examples=train_examples, tokenizer=tokenizer, max_seq_length=max_seq_length,
        doc_stride=doc_stride, max_query_length=max_query_length, is_training=True,
        entity_linker=entity_linker, entity_vocab=entity_vocab, max_entity_length=max_entity_length,
        max_mention_length=max_mention_length, use_entities=use_entities)

    all_word_ids = torch.tensor(
        [f.word_ids for f in train_features], dtype=torch.long)
    all_word_attention_mask = torch.tensor(
        [f.word_attention_mask for f in train_features], dtype=torch.long)
    all_word_segment_ids = torch.tensor(
        [f.word_segment_ids for f in train_features], dtype=torch.long)
    all_entity_ids = torch.tensor(
        [f.entity_ids for f in train_features], dtype=torch.long)
    all_entity_position_ids = torch.tensor(
        [f.entity_position_ids for f in train_features], dtype=torch.long)
    all_entity_segment_ids = torch.tensor(
        [f.entity_segment_ids for f in train_features], dtype=torch.long)
    all_entity_attention_mask = torch.tensor(
        [f.entity_attention_mask for f in train_features], dtype=torch.long)

    all_start_positions = torch.tensor(
        [f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor(
        [f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_word_ids, all_word_attention_mask, all_word_segment_ids,
                               all_entity_ids, all_entity_position_ids, all_entity_segment_ids,
                               all_entity_attention_mask, all_start_positions, all_end_positions)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=train_batch_size)

    eval_examples = read_squad_examples(
        input_file=dev_file_path, is_training=False, version_2_with_negative=version_2_with_negative)

    eval_features = convert_examples_to_features(
        examples=eval_examples, tokenizer=tokenizer, max_seq_length=max_seq_length,
        doc_stride=doc_stride, max_query_length=max_query_length, is_training=False,
        entity_linker=entity_linker, entity_vocab=entity_vocab, max_entity_length=max_entity_length,
        max_mention_length=max_mention_length, use_entities=use_entities)

    all_word_ids = torch.tensor(
        [f.word_ids for f in eval_features], dtype=torch.long)
    all_word_attention_mask = torch.tensor(
        [f.word_attention_mask for f in eval_features], dtype=torch.long)
    all_word_segment_ids = torch.tensor(
        [f.word_segment_ids for f in eval_features], dtype=torch.long)
    all_entity_ids = torch.tensor(
        [f.entity_ids for f in eval_features], dtype=torch.long)
    all_entity_position_ids = torch.tensor(
        [f.entity_position_ids for f in eval_features], dtype=torch.long)
    all_entity_segment_ids = torch.tensor(
        [f.entity_segment_ids for f in eval_features], dtype=torch.long)
    all_entity_attention_mask = torch.tensor(
        [f.entity_attention_mask for f in eval_features], dtype=torch.long)

    all_example_index = torch.arange(all_word_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(
        all_word_ids, all_word_attention_mask, all_word_segment_ids,
        all_entity_ids, all_entity_position_ids, all_entity_segment_ids,
        all_entity_attention_mask, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    for _ in trange(int(iteration), desc="Epoch"):
        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for (step, batch) in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            (word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
             entity_segment_ids, entity_attention_mask, start_position, end_position) = batch
            max_word_size = max(
                int(torch.max(torch.sum(word_attention_mask, dim=1)).item()), 1)
            max_entity_size = max(
                int(torch.max(torch.sum(entity_attention_mask, dim=1)).item()), 1)
            loss = model(word_ids=word_ids[:, :max_word_size],
                         word_segment_ids=word_segment_ids[:, :max_word_size],
                         word_attention_mask=word_attention_mask[:,
                                                                 :max_word_size],
                         entity_ids=entity_ids[:, :max_entity_size],
                         entity_position_ids=entity_position_ids[:,
                                                                 :max_entity_size],
                         entity_segment_ids=entity_segment_ids[:,
                                                               :max_entity_size],
                         entity_attention_mask=entity_attention_mask[:,
                                                                     :max_entity_size],
                         start_positions=start_position,
                         end_positions=start_position)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += word_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

        model.eval()
        all_results = []

        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            (word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
             entity_segment_ids, entity_attention_mask, example_indices) = batch
            max_word_size = max(
                int(torch.max(torch.sum(word_attention_mask, dim=1)).item()), 1)
            max_entity_size = max(
                int(torch.max(torch.sum(entity_attention_mask, dim=1)).item()), 1)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(word_ids=word_ids[:, :max_word_size],
                                                             word_segment_ids=word_segment_ids[:,
                                                                                               :max_word_size],
                                                             word_attention_mask=word_attention_mask[:,
                                                                                                     :max_word_size],
                                                             entity_ids=entity_ids[:,
                                                                                   :max_entity_size],
                                                             entity_position_ids=entity_position_ids[:,
                                                                                                     :max_entity_size],
                                                             entity_segment_ids=entity_segment_ids[:,
                                                                                                   :max_entity_size],
                                                             entity_attention_mask=entity_attention_mask[:,
                                                                                                         :max_entity_size])

            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))

        output_prediction_file = os.path.join(
            output_dir, "predictions.json")
        output_nbest_file = os.path.join(
            output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(
            output_dir, "null_odds.json")

        write_predictions(all_examples=eval_examples, all_features=eval_features, all_results=all_results,
                          n_best_size=n_best_size, max_answer_length=max_answer_length, cased=cased,
                          output_prediction_file=output_prediction_file,
                          output_nbest_file=output_nbest_file,
                          output_null_log_odds_file=output_null_log_odds_file,
                          version_2_with_negative=version_2_with_negative,
                          null_score_diff_threshold=null_score_diff_threshold)


if __name__ == '__main__':
    run()
