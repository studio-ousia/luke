# -*- coding: utf-8 -*-

import collections
import inspect
import json
import logging
import os
import random
import joblib
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from batch_generator import create_word_data
from model import LukeModel, LukeConfig, EntityPredictionHead
from model_common import LayerNorm
from optimization import BertAdam
from utils import clean_text
from utils.aida_conll import AIDACoNLLReader
from utils.vocab import EntityVocab, MASK_TOKEN
from wiki_corpus import WikiCorpus

logger = logging.getLogger(__name__)


class LukeForEntityDisambiguation(LukeModel):
    def __init__(self, config):
        super(LukeForEntityDisambiguation, self).__init__(config)

        self.entity_predictions = EntityPredictionHead(config,
            self.entity_embeddings.entity_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self, word_ids, word_segment_ids, word_attention_mask, entity_ids,
                entity_position_ids, entity_segment_ids, entity_attention_mask,
                entity_candidate_ids, entity_label=None):
        (encoded_layers, _) = super(LukeForEntityDisambiguation, self).forward(
            word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
            entity_segment_ids, entity_attention_mask, output_all_encoded_layers=False)

        entity_sequence_output = encoded_layers[1]

        logits = self.entity_predictions(entity_sequence_output).view(
            -1, self.config.entity_vocab_size)
        entity_candidate_mask = logits.new_full(logits.size(), 0, dtype=torch.uint8)
        entity_candidate_mask.scatter_(dim=1, index=entity_candidate_ids,
                                       src=(entity_candidate_ids != 0))
        masked_logits = logits.masked_fill((1 - entity_candidate_mask), -1e32)

        if entity_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(masked_logits, entity_label)
            return loss
        else:
            return masked_logits


def process_documents(documents, dump_db, mention_db, tokenizer, entity_vocab, max_seq_length,
                      max_candidate_size):
    ret = []
    max_num_tokens = max_seq_length - 2
    for document in tqdm(documents):
        orig_words = [clean_text(w) for w in document.words]
        mention_start_map = {m.span[0]: m for m in document.mentions}
        mention_end_map = {m.span[1]: m for m in document.mentions}

        tokens = []
        for (orig_pos, orig_word) in enumerate(orig_words):
            if orig_pos in mention_start_map:
                mention_start_map[orig_pos].span = [len(tokens), None]

            if orig_pos in mention_end_map:
                mention_end_map[orig_pos].span[1] = len(tokens)

            tokens.extend(tokenizer.tokenize(orig_word))

        entity_ids = np.array([entity_vocab[MASK_TOKEN]], dtype=np.int)
        entity_segment_ids = np.array([0], dtype=np.int)
        entity_attention_mask = np.array([1], dtype=np.int)

        for mention in document.mentions:
            title = dump_db.resolve_redirect(mention.title)
            span = mention.span

            mention_size = span[1] - span[0]
            half_cxt_size = int((max_num_tokens - mention_size) / 2)

            left_token_size = span[0]
            right_token_size = len(tokens) - span[1]
            if left_token_size < right_token_size:
                left_context_size = min(left_token_size, half_cxt_size)
                right_context_size = min(right_token_size,
                                         max_num_tokens - left_context_size - mention_size)
            else:
                right_context_size = min(right_token_size, half_cxt_size)
                left_context_size = min(left_token_size,
                                        max_num_tokens - right_context_size - mention_size)

            target_tokens = tokens[span[0] - left_context_size:span[0]]
            target_tokens += tokens[span[0]:span[1]]
            target_tokens += tokens[span[1]:span[1] + right_context_size]

            # print('left', tokens[span[0] - left_context_size:span[0]])
            # print('mention', tokens[span[0]:span[1]])
            # print('right', tokens[span[1]:span[1] + right_context_size])

            word_data = create_word_data(target_tokens, None, tokenizer.vocab, max_seq_length)

            entity_position_ids = np.array([left_context_size + 1], dtype=np.int)  # 1 for CLS
            entity_candidate_ids = np.zeros(max_candidate_size, dtype=np.int)
            # entity_candidate_mask = np.zeros(len(entity_vocab), dtype=np.int)

            try:
                candidate_data = mention_db.query(clean_text(mention.text))
                prior_probs = {c.title: c.prior_prob for c in candidate_data}
            except KeyError:
                prior_probs = {}

            candidates = [dump_db.resolve_redirect(c) for c in mention.candidates]
            candidates = sorted(candidates,
                                key=lambda c: prior_probs.get(c, 0.0))[-max_candidate_size:]
            candidates = [entity_vocab[c] for c in candidates if c in entity_vocab]
            entity_candidate_ids[:len(candidates)] = candidates

            if title in entity_vocab:
                entity_label = entity_vocab[title]
            else:
                entity_label = -1

            feature = dict(word_ids=word_data['word_ids'],
                           word_segment_ids=word_data['word_segment_ids'],
                           word_attention_mask=word_data['word_attention_mask'],
                           entity_ids=entity_ids,
                           entity_position_ids=entity_position_ids,
                           entity_segment_ids=entity_segment_ids,
                           entity_attention_mask=entity_attention_mask,
                           entity_candidate_ids=entity_candidate_ids,
                           entity_label=entity_label)

            ret.append((document, mention, feature))

    return ret


def run(data_dir, dump_db, mention_db, model_file, output_dir, max_seq_length, max_candidate_size,
        batch_size, eval_batch_size, learning_rate, iteration, warmup_proportion, lr_decay, seed,
        gradient_accumulation_steps, fix_entity_emb):
    device = torch.device('cuda')
    n_gpu = torch.cuda.device_count()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data_file = model_file.replace('.bin', '.pkl').replace('model', 'data')
    model_data = joblib.load(data_file)

    corpus = WikiCorpus(model_data['args']['corpus_data_file'])
    entity_vocab = EntityVocab(model_data['args']['entity_vocab_file'])

    train_batch_size = int(batch_size / gradient_accumulation_steps)

    config = LukeConfig(**model_data['config'])

    model = LukeForEntityDisambiguation(config)
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    if fix_entity_emb:
        model.entity_embeddings.entity_embeddings.weight.requires_grad = False

    forward_arguments = inspect.getfullargspec(LukeForEntityDisambiguation.forward)[0][1:]

    dataset = AIDACoNLLReader(data_dir)
    train_data = process_documents(dataset.get_documents('train'), dump_db, mention_db,
        corpus.tokenizer, entity_vocab, max_seq_length, max_candidate_size)
    train_tensors = TensorDataset(*[torch.tensor([f[k] for (_, _, f) in train_data], dtype=torch.long)
                                    for k in forward_arguments])
    train_sampler = RandomSampler(train_tensors)
    train_dataloader = DataLoader(train_tensors, sampler=train_sampler, batch_size=train_batch_size)

    eval_data = process_documents(dataset.get_documents('test'), dump_db, mention_db,
        corpus.tokenizer, entity_vocab, max_seq_length, max_candidate_size)
    eval_tensors = TensorDataset(*[torch.tensor([f[k] for (_, _, f) in eval_data], dtype=torch.long)
                                   for k in forward_arguments])
    eval_sampler = SequentialSampler(eval_tensors)
    eval_dataloader = DataLoader(eval_tensors, sampler=eval_sampler, batch_size=eval_batch_size)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    num_train_steps = int(len(train_tensors) / batch_size * iteration)

    parameters = {'params': [], 'weight_decay': 0.01}
    no_decay_parameters = {'params': [], 'weight_decay': 0.0}

    for module in model.modules():
        if isinstance(module, LayerNorm):
            no_decay_parameters['params'].extend(list(module.parameters(recurse=False)))
        else:
            for (name, param) in module.named_parameters(recurse=False):
                if 'bias' in name:
                    no_decay_parameters['params'].append(param)
                else:
                    parameters['params'].append(param)

    opt_device = torch.device('cuda:0')
    optimizer = BertAdam([parameters, no_decay_parameters], lr=learning_rate, device=opt_device,
                         warmup=warmup_proportion, lr_decay=lr_decay, t_total=num_train_steps)
    global_step = 0

    if output_dir:
        writer = open(os.path.join(output_dir, 'eval_results.jl'), 'a')

    for n_iter in trange(int(iteration), desc='Epoch'):
        model.eval()

        eval_loss = 0
        eval_logits = []
        eval_labels = []
        nb_eval_steps = 0
        for batch in tqdm(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                tmp_eval_loss = model(*batch)
                logits = model(*batch[:-1])

            eval_logits.append(logits.detach().cpu().numpy())
            eval_labels.append(batch[-1].cpu().numpy())

            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        eval_labels = np.concatenate(eval_labels)
        outputs = np.argmax(np.vstack(eval_logits), axis=1)

        num_correct = 0
        num_total = 0
        num_doc_correct = collections.defaultdict(int)
        num_doc_total = collections.defaultdict(int)
        for (predicted, correct, (document, mention, _)) in zip(outputs, eval_labels, eval_data):
            if predicted == correct:
                # print(predicted, correct)
                num_correct += 1
                num_doc_correct[document.id] += 1
            elif len(mention.candidates) == 1:  # unambiguous mention
                num_correct += 1
                num_doc_correct[document.id] += 1
            num_total += 1
            num_doc_total[document.id] += 1

        macro_acc = np.mean([num_doc_correct[d] / num_doc_total[d] for d in num_doc_correct.keys()])
        micro_acc = num_correct / num_total

        result = dict(
            eval_loss=eval_loss,
            global_step=global_step,
            # loss=tr_loss / nb_tr_steps,
            model_file=model_file,
            batch_size=batch_size,
            learning_rate=learning_rate,
            iteration=n_iter,
            total_iteration=iteration,
            warmup_proportion=warmup_proportion,
            seed=seed,
            lr_decay=lr_decay,
            max_seq_length=max_seq_length,
        )

        result['eval_micro_accuracy'] = micro_acc
        result['eval_macro_accuracy'] = macro_acc

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            if key.startswith('eval_') or key in ('loss', 'global_step', 'iteration'):
                logger.info("  %s = %s", key, str(result[key]))
        if output_dir:
            writer.write("%s\n" % json.dumps(result, sort_keys=True))
            writer.flush()

        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for (step, batch) in enumerate(tqdm(train_dataloader, desc='Iteration')):
            batch = tuple(t.to(device) for t in batch)
            loss = model(*batch)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += batch[0].size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

    if output_dir:
        writer.close()
