import copy
import inspect
import json
import logging
import os
import pickle
import random
import click
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from luke.batch_generator import create_word_data
from luke.model import LukeConfig
from luke.optimization import BertAdam
from luke.utils import clean_text
from luke.utils.vocab import WordPieceVocab, EntityVocab, MASK_TOKEN, PAD_TOKEN
from luke.utils.word_tokenizer import WordPieceTokenizer

from ed_dataset import EntityDisambiguationDataset
from ed_model import LukeForEntityDisambiguation

DATASET_CACHE_FILE = 'entity_disambiguation_dataset.pkl'

logger = logging.getLogger(__name__)


@click.command()
@click.argument('word_vocab_file', type=click.Path(exists=True))
@click.argument('entity_vocab_file', type=click.Path(exists=True))
@click.argument('wikipedia_titles_file', type=click.Path(exists=True))
@click.argument('wikipedia_redirects_file', type=click.Path(exists=True))
@click.argument('model_file', type=click.Path())
@click.option('-v', '--verbose', is_flag=True)
@click.option('--data-dir', type=click.Path(exists=True), default='data/entity-disambiguation')
@click.option('--cased/--uncased', default=False)
@click.option('--max-seq-length', default=512)
@click.option('--batch-size', default=32)
@click.option('--learning-rate', default=1e-5)
@click.option('--iteration', default=3.0)
@click.option('--eval-batch-size', default=8)
@click.option('--warmup-proportion', default=0.1)
@click.option('--lr-decay/--no-lr-decay', default=True)
@click.option('--seed', default=42)
@click.option('--gradient-accumulation-steps', default=32)
@click.option('--max-entity-length', default=128)
@click.option('--max-candidate-size', default=30)
@click.option('--max-mention-length', default=20)
@click.option('--min-context-prior-prob', default=0.7)
@click.option('--fix-word-emb/--update-word-emb', default=False)
@click.option('--fix-entity-emb/--update-entity-emb', default=True)
@click.option('--fix-entity-bias/--update-entity-bias', default=True)
@click.option('--evaluate-every-epoch', is_flag=True)
@click.option('--in-domain/--out-domain', default=True)
@click.option('-t', '--test-set', default=None, multiple=True)
def run(data_dir, word_vocab_file, entity_vocab_file, wikipedia_titles_file, wikipedia_redirects_file,
        model_file, verbose, cased, max_seq_length, max_entity_length, max_candidate_size,
        max_mention_length, min_context_prior_prob, batch_size, eval_batch_size, learning_rate,
        iteration, warmup_proportion, lr_decay, seed, gradient_accumulation_steps, fix_word_emb,
        fix_entity_emb, fix_entity_bias, evaluate_every_epoch, in_domain, test_set):
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if not test_set:
        if in_domain:
            test_set = ['test_b']
        else:
            test_set = ['ace2004', 'aquaint', 'msnbc', 'wikipedia', 'clueweb']

    logger.info('Loading model and configurations...')

    state_dict = torch.load(model_file + '.bin', map_location='cpu')
    json_file = model_file + '.json'
    with open(json_file) as f:
        model_data = json.load(f)

    config = LukeConfig(**model_data['model_config'])
    word_vocab = WordPieceVocab(word_vocab_file)
    tokenizer = WordPieceTokenizer(word_vocab, not cased)
    orig_entity_vocab = EntityVocab(entity_vocab_file, 'tsv')

    logger.info('Loading dataset...')

    if os.path.exists(DATASET_CACHE_FILE):
        logger.info('Using cache: %s', DATASET_CACHE_FILE)
        with open(DATASET_CACHE_FILE, mode='rb') as f:
            (dataset, entity_titles) = pickle.load(f)
    else:
        dataset = EntityDisambiguationDataset(data_dir)
        with open(wikipedia_titles_file) as f:
            valid_titles = frozenset([l.rstrip() for l in f])
        redirects = {}
        with open(wikipedia_redirects_file) as f:
            for line in f:
                (src, dest) = line.rstrip().split('\t')
                redirects[src] = dest

        # build entity vocabulary and resolve Wikipedia redirects
        entity_titles = set([MASK_TOKEN])
        for documents in dataset.get_all_datasets():
            for document in documents:
                new_mentions = []
                for mention in document.mentions:
                    mention.title = redirects.get(mention.title, mention.title)
                    if mention.title not in valid_titles:
                        logger.debug('Invalid title: %s', mention.title)
                        continue
                    entity_titles.add(mention.title)
                    new_mentions.append(mention)
                    for candidate in mention.candidates:
                        candidate.title = redirects.get(candidate.title, candidate.title)
                        entity_titles.add(candidate.title)
                document.mentions = new_mentions

        with open(DATASET_CACHE_FILE, mode='wb') as f:
            pickle.dump((dataset, entity_titles), f)

    # build a vocabulary, embeddings and biases of entities contained in the dataset
    orig_entity_emb = state_dict['entity_embeddings.entity_embeddings.weight']
    orig_entity_bias = state_dict['entity_predictions.bias']
    entity_emb = orig_entity_emb.new_zeros((len(entity_titles) + 1, config.entity_emb_size))
    entity_bias = orig_entity_bias.new_zeros(len(entity_titles) + 1)
    entity_vocab = {PAD_TOKEN: 0}
    for (n, title) in enumerate(entity_titles, 1):
        if title in orig_entity_vocab:
            entity_vocab[title] = n
            orig_index = orig_entity_vocab[title]
            entity_emb[n] = orig_entity_emb[orig_index]
            entity_bias[n] = orig_entity_bias[orig_index]
        else:
            entity_vocab[title] = n

    config.entity_vocab_size = len(entity_vocab)
    state_dict['entity_embeddings.entity_embeddings.weight'] = entity_emb
    state_dict['entity_predictions.decoder.weight'] = entity_emb
    state_dict['entity_predictions.bias'] = entity_bias

    logger.info('Model configuration: %s', config)

    model = LukeForEntityDisambiguation(config)
    model.load_state_dict(state_dict, strict=False)
    model.to('cuda')

    model_arg_names = inspect.getfullargspec(LukeForEntityDisambiguation.forward)[0][1:]

    def evaluate(model, dataset_name):
        model.eval()

        documents = getattr(dataset, dataset_name)
        eval_data = generate_features(documents, tokenizer, entity_vocab, max_seq_length,
                                      max_entity_length, max_candidate_size, min_context_prior_prob,
                                      max_mention_length)
        eval_tensors = TensorDataset(*[torch.tensor([f[k] for (_, _, f) in eval_data],
                                       dtype=torch.long) for k in model_arg_names])
        eval_dataloader = DataLoader(eval_tensors, sampler=SequentialSampler(eval_tensors),
                                     batch_size=eval_batch_size)

        (precision, recall, f1) = compute_precision_recall_f1(model, eval_data, eval_dataloader,
                                                              dataset_name)

        logger.info("***** Eval results: %s *****", dataset_name)
        logger.info("  F1 = %.3f", f1)
        logger.info("  precision = %.3f", precision)
        logger.info("  recall = %.3f", recall)

        return (precision, recall, f1)

    n_iter = 0
    results = []

    if in_domain:
        logger.info('Fix word embeddings during training: %s', fix_word_emb)
        if fix_word_emb:
            model.embeddings.word_embeddings.weight.requires_grad = False

        logger.info('Fix entity embeddings during training: %s', fix_entity_emb)
        if fix_entity_emb:
            model.entity_embeddings.entity_embeddings.weight.requires_grad = False

        logger.info('Fix entity bias during training: %s', fix_entity_bias)
        if fix_entity_bias:
            model.entity_predictions.bias.requires_grad = False

        logger.info('Creating TensorDataset for training...')

        train_batch_size = int(batch_size / gradient_accumulation_steps)

        train_data = generate_features(dataset.train, tokenizer, entity_vocab, max_seq_length,
            max_entity_length, max_candidate_size, min_context_prior_prob, max_mention_length)
        train_tensors = TensorDataset(*[torch.tensor([f[k] for (_, _, f) in train_data], dtype=torch.long)
                                        for k in model_arg_names])
        train_dataloader = DataLoader(train_tensors, sampler=RandomSampler(train_tensors),
                                      batch_size=train_batch_size)

        num_train_steps = int(len(train_tensors) / batch_size * iteration)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_parameters, lr=learning_rate, warmup=warmup_proportion,
                             lr_decay=lr_decay, t_total=num_train_steps)

        for n_iter in range(int(iteration)):
            if evaluate_every_epoch:
                for dataset_name in test_set:
                    results.append(evaluate(model, dataset_name))

            model.train()
            logger.info("***** Epoch: %d/%d *****", n_iter + 1, iteration)

            for (step, batch) in enumerate(tqdm(train_dataloader, desc='train')):
                batch = tuple(t.to('cuda') for t in batch)
                loss = model(*batch)
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()

    for dataset_name in test_set:
        results.append(evaluate(model, dataset_name))

    return results


def generate_features(documents, tokenizer, entity_vocab, max_seq_length, max_entity_length,
                      max_candidate_size, min_context_prior_prob, max_mention_length):
    ret = []
    max_num_tokens = max_seq_length - 2
    for document in documents:
        document = copy.deepcopy(document)
        orig_words = [clean_text(w) for w in document.words]
        mention_start_map = {m.span[0]: m for m in document.mentions}
        mention_end_map = {m.span[1]: m for m in document.mentions}

        tokens = []
        for (orig_pos, orig_word) in enumerate(orig_words):
            if orig_pos in mention_start_map:
                mention_start_map[orig_pos].start = len(tokens)

            if orig_pos in mention_end_map:
                mention_end_map[orig_pos].end = len(tokens)

            tokens.extend(tokenizer.tokenize(orig_word))

        for target_mention in document.mentions:
            target_span = target_mention.span

            mention_length = target_span[1] - target_span[0]
            half_context_size = int((max_num_tokens - mention_length) / 2)

            left_token_size = target_span[0]
            right_token_size = len(tokens) - target_span[1]
            if left_token_size < right_token_size:
                left_context_size = min(left_token_size, half_context_size)
                right_context_size = min(right_token_size,
                                         max_num_tokens - left_context_size - mention_length)
            else:
                right_context_size = min(right_token_size, half_context_size)
                left_context_size = min(left_token_size,
                                        max_num_tokens - right_context_size - mention_length)

            token_start = target_span[0] - left_context_size
            token_end = target_span[1] + right_context_size
            target_tokens = tokens[token_start:target_span[0]]
            target_tokens += tokens[target_span[0]:target_span[1]]
            target_tokens += tokens[target_span[1]:token_end]

            word_data = create_word_data(target_tokens, None, tokenizer.vocab, max_seq_length)

            entity_ids = np.zeros(max_entity_length, dtype=np.int)
            entity_ids[0] = entity_vocab[MASK_TOKEN]

            entity_position_ids = np.full((max_entity_length, max_mention_length), -1, dtype=np.int)
            entity_position_ids[0][:mention_length] = range(left_context_size + 1,
                left_context_size + mention_length + 1)  # +1 for [CLS]

            entity_index = 1
            for mention in document.mentions:
                if mention == target_mention:
                    continue

                if entity_index == max_entity_length:
                    break

                start = mention.start - token_start
                end = mention.end - token_start

                if start < 0 or end > max_num_tokens:
                    continue

                mention_length = end - start
                for candidate in mention.candidates:
                    if candidate.prior_prob <= min_context_prior_prob:
                        continue
                    entity_ids[entity_index] = entity_vocab[candidate.title]
                    entity_position_ids[entity_index][:mention_length] = range(start + 1, end + 1)  # +1 for [CLS]
                    entity_index += 1
                    if entity_index == max_entity_length:
                        break

            entity_segment_ids = np.zeros(max_entity_length, dtype=np.int)
            entity_attention_mask = np.zeros(max_entity_length, dtype=np.int)
            entity_attention_mask[:entity_index] = 1

            entity_candidate_ids = np.zeros(max_candidate_size, dtype=np.int)

            candidates = target_mention.candidates[:max_candidate_size]
            entity_candidate_ids[:len(candidates)] = [entity_vocab[c.title] for c in candidates]

            entity_label = entity_vocab[target_mention.title]

            feature = dict(word_ids=word_data['word_ids'],
                           word_segment_ids=word_data['word_segment_ids'],
                           word_attention_mask=word_data['word_attention_mask'],
                           entity_ids=entity_ids,
                           entity_position_ids=entity_position_ids,
                           entity_segment_ids=entity_segment_ids,
                           entity_attention_mask=entity_attention_mask,
                           entity_candidate_ids=entity_candidate_ids,
                           entity_label=entity_label)

            ret.append((document, target_mention, feature))

    return ret


def compute_precision_recall_f1(model, eval_data, eval_dataloader, desc):
    eval_logits = []
    eval_labels = []
    for batch in tqdm(eval_dataloader, desc=desc, leave=False):
        args = [t.to('cuda') for t in batch[:-1]]
        with torch.no_grad():
            logits = model(*args)

        eval_logits.append(logits.detach().cpu().numpy())
        eval_labels.append(batch[-1].numpy())

    eval_labels = np.concatenate(eval_labels)
    outputs = np.argmax(np.vstack(eval_logits), axis=1)

    num_correct = 0
    num_mentions = 0
    num_mentions_with_candidates = 0
    for (predicted, correct, (_, mention, _)) in zip(outputs, eval_labels, eval_data):
        if predicted == correct:
            num_correct += 1

        assert correct != 0

        num_mentions += 1
        if mention.candidates:
            num_mentions_with_candidates += 1

    logger.info('#mentions (%s): %d', desc, num_mentions)
    logger.info('#mentions with candidates (%s): %d', desc, num_mentions_with_candidates)
    logger.info('#correct (%s): %d', desc, num_correct)

    precision = num_correct / num_mentions_with_candidates
    recall = num_correct / num_mentions
    f1 = 2.0 * precision * recall / (precision + recall)

    return (precision, recall, f1)


if __name__ == '__main__':
    run()
