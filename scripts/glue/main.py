# -*- coding: utf-8 -*-

import json
import os
import logging
import random
import click
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm, trange

from luke.optimization import BertAdam
from luke.model import LukeConfig
from luke.utils.entity_linker import EntityLinker, MentionDB
from luke.utils.vocab import WordPieceVocab, EntityVocab
from luke.utils.word_tokenizer import WordPieceTokenizer
from glue_dataset import ColaProcessor, MnliProcessor, QnliProcessor, MrpcProcessor, QqpProcessor,\
    SciTailProcessor, RTEProcessor, STSProcessor, convert_examples_to_features
from glue_model import LukeForSequenceClassification

logger = logging.getLogger(__name__)


@click.command()
@click.argument('word_vocab_file', type=click.Path(exists=True))
@click.argument('entity_vocab_file', type=click.Path(exists=True))
@click.argument('mention_db_file', type=click.Path(exists=True))
@click.argument('model_file', type=click.Path())
@click.option('--data-dir', type=click.Path(exists=True), default='data')
@click.option('--cased/--uncased', default=False)
@click.option('-t', '--task-name', default='mrpc')
@click.option('--max-seq-length', default=512)
@click.option('--max-entity-length', default=128)
@click.option('--max-mention-length', default=20)
@click.option('--batch-size', default=32)
@click.option('--eval-batch-size', default=8)
@click.option('--learning-rate', default=1e-5)
@click.option('--iteration', default=3.0)
@click.option('--warmup-proportion', default=0.1)
@click.option('--lr-decay/--no-lr-decay', default=True)
@click.option('--seed', default=42)
@click.option('--gradient-accumulation-steps', default=32)
@click.option('--fix-entity-emb/--update-entity-emb', default=True)
@click.option('--use-entities/--no-entities', default=True)
@click.option('--min-prior-prob', default=0.1)
def run(word_vocab_file, entity_vocab_file, mention_db_file, model_file, data_dir, cased, task_name,
        max_seq_length, max_entity_length, max_mention_length, batch_size, eval_batch_size,
        learning_rate, iteration, warmup_proportion, lr_decay, seed, gradient_accumulation_steps,
        fix_entity_emb, use_entities, min_prior_prob):
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

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "qnli": QnliProcessor,
        "mrpc": MrpcProcessor,
        "qqp": QqpProcessor,
        "scitail": SciTailProcessor,
        "rte": RTEProcessor,
        "sts-b": STSProcessor,
    }
    processor = processors[task_name]()
    data_dir = os.path.join(data_dir, task_name)

    json_file = model_file + '.json'
    with open(json_file) as f:
        model_data = json.load(f)

    model_config = model_data['model_config']
    config = LukeConfig(**model_config)
    logger.info('Model configuration: %s', config)

    if processor.task_type == 'classification':
        label_list = processor.get_labels()
        label_dtype = torch.long
        model = LukeForSequenceClassification(config, len(label_list))

    else:  # regression
        label_list = None  # scoring task
        label_dtype = torch.float
        # model = LukeForSequenceRegression(config)

    state_dict = torch.load(model_file + '.bin', map_location='cpu')
    model_state_dict = model.state_dict()
    model_state_dict.update({k: v for k, v in state_dict.items() if k in model_state_dict})
    model.load_state_dict(model_state_dict)
    # model.load_state_dict(state_dict, strict=False)
    del state_dict, model_state_dict

    logger.info('Fix entity embeddings during training: %s', fix_entity_emb)
    model.embeddings.word_embeddings.sparse = True
    model.entity_embeddings.entity_embeddings.sparse = True
    if fix_entity_emb:
        model.entity_embeddings.entity_embeddings.weight.requires_grad = False

    device = torch.device("cuda")
    model.to(device)

    train_batch_size = int(batch_size / gradient_accumulation_steps)

    train_examples = processor.get_train_examples(data_dir)
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
        train_examples, processor.task_type, label_list, tokenizer, entity_linker, entity_vocab,
        max_seq_length, max_entity_length, max_mention_length, use_entities)

    train_data = TensorDataset(
        torch.tensor([f.word_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.word_segment_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.word_attention_mask for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_position_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_segment_ids for f in train_features], dtype=torch.long),
        torch.tensor([f.entity_attention_mask for f in train_features], dtype=torch.long),
        torch.tensor([f.label for f in train_features], dtype=label_dtype),
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    eval_examples = processor.get_dev_examples(data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, processor.task_type, label_list, tokenizer, entity_linker, entity_vocab,
        max_seq_length, max_entity_length, max_mention_length, use_entities)

    eval_data = TensorDataset(
        torch.tensor([f.word_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.word_segment_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.word_attention_mask for f in eval_features], dtype=torch.long),
        torch.tensor([f.entity_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.entity_position_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.entity_segment_ids for f in eval_features], dtype=torch.long),
        torch.tensor([f.entity_attention_mask for f in eval_features], dtype=torch.long),
        torch.tensor([f.label for f in eval_features], dtype=label_dtype),
    )
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    for _ in trange(int(iteration), desc="Epoch"):
        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for (step, batch) in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            (word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
             entity_segment_ids, entity_attention_mask, labels) = batch
            max_word_size = max(int(torch.max(torch.sum(word_attention_mask, dim=1)).item()), 1)
            max_entity_size = max(int(torch.max(torch.sum(entity_attention_mask, dim=1)).item()), 1)
            loss = model(word_ids=word_ids[:, :max_word_size],
                         word_segment_ids=word_segment_ids[:, :max_word_size],
                         word_attention_mask=word_attention_mask[:, :max_word_size],
                         entity_ids=entity_ids[:, :max_entity_size],
                         entity_position_ids=entity_position_ids[:, :max_entity_size],
                         entity_segment_ids=entity_segment_ids[:, :max_entity_size],
                         entity_attention_mask=entity_attention_mask[:, :max_entity_size],
                         labels=labels)
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

        eval_logits = []
        eval_labels = []
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            (word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
             entity_segment_ids, entity_attention_mask, labels) = batch
            max_word_size = max(int(torch.max(torch.sum(word_attention_mask, dim=1)).item()), 1)
            max_entity_size = max(int(torch.max(torch.sum(entity_attention_mask, dim=1)).item()), 1)
            with torch.no_grad():
                logits = model(word_ids=word_ids[:, :max_word_size],
                               word_segment_ids=word_segment_ids[:, :max_word_size],
                               word_attention_mask=word_attention_mask[:, :max_word_size],
                               entity_ids=entity_ids[:, :max_entity_size],
                               entity_position_ids=entity_position_ids[:, :max_entity_size],
                               entity_segment_ids=entity_segment_ids[:, :max_entity_size],
                               entity_attention_mask=entity_attention_mask[:, :max_entity_size])

            eval_logits.append(logits.detach().cpu().numpy())
            eval_labels.append(labels.cpu().numpy())

            # eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += batch[0].size(0)
            nb_eval_steps += 1

        # eval_loss = eval_loss / nb_eval_steps

        result = dict(train_loss=tr_loss / nb_tr_steps)

        if processor.task_type == 'classification':
            outputs = np.argmax(np.vstack(eval_logits), axis=1)
            result['eval_accuracy'] = np.sum(outputs == np.concatenate(eval_labels)) / nb_eval_examples
        elif processor.task_type == 'regression':
            outputs = np.vstack(eval_logits).flatten()
            result['eval_pearson'] = float(pearsonr(outputs, np.concatenate(eval_labels))[0])
            result['eval_spearman'] = float(spearmanr(outputs, np.concatenate(eval_labels))[0])

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))


if __name__ == '__main__':
    run()
