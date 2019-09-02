import json
import logging
import math
import os
import random
import click
from comet_ml import Experiment
import numpy as np
from pytorch_transformers.tokenization_bert import BertTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm

from luke.model import LukeConfig
from luke.optimization import LukeDenseSparseAdam, WarmupInverseSquareRootSchedule
from luke.utils.entity_vocab import EntityVocab, MASK_TOKEN, PAD_TOKEN

from dataset import EntityDisambiguationDataset, generate_data
from model import LukeForEntityDisambiguation

logger = logging.getLogger(__name__)


@click.command()
@click.argument('wikipedia_titles_file', type=click.Path(exists=True))
@click.argument('wikipedia_redirects_file', type=click.Path(exists=True))
@click.argument('model_file', type=click.Path())
@click.option('-v', '--verbose', is_flag=True)
@click.option('--data-dir', type=click.Path(exists=True), default='data/entity-disambiguation')
@click.option('--max-candidate-length', default=30)
@click.option('--num-documents-per-batch', default=32)
@click.option('--document-split-mode', default='simple', type=click.Choice(['simple', 'per_mention']))
@click.option('--min-context-entity-prob', default=0.0)
@click.option('--learning-rate', default=1e-5)
@click.option('--patience', default=5)
@click.option('--warmup-steps', default=50)
@click.option('--grad-avg-on-cpu', is_flag=True)
@click.option('--seed', default=42)
@click.option('--fix-entity-emb/--update-entity-emb', default=True)
@click.option('--fix-entity-bias/--update-entity-bias', default=True)
@click.option('--in-domain/--out-domain', default=True)
@click.option('-t', '--test-set', default=None, multiple=True)
def run(data_dir, wikipedia_titles_file, wikipedia_redirects_file, model_file, verbose, max_candidate_length,
        num_documents_per_batch, document_split_mode, min_context_entity_prob, learning_rate, patience, warmup_steps,
        grad_avg_on_cpu, seed, fix_entity_emb, fix_entity_bias, in_domain, test_set):
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

    if in_domain:
        experiment = Experiment(project_name='luke_entity_disambiguation_in_domain')
    else:
        experiment = Experiment(project_name='luke_entity_disambiguation_out_domain')

    experiment.log_parameters(dict(
        model_file=model_file,
        num_documents_per_batch=num_documents_per_batch,
        document_split_mode=document_split_mode,
        min_context_entity_prob=min_context_entity_prob,
        learning_rate=learning_rate,
        patience=patience,
        warmup_steps=warmup_steps,
        seed=seed,
        fix_entity_emb=fix_entity_emb,
        fix_entity_bias=fix_entity_bias,
    ))

    logger.info('Loading model and configurations...')

    model_dir = os.path.dirname(model_file)
    state_dict = torch.load(model_file, map_location='cpu')
    json_file = os.path.join(model_dir, 'metadata.json')
    with open(json_file) as f:
        model_data = json.load(f)

    config = LukeConfig(**model_data['model_config'])
    max_seq_length = model_data['max_seq_length']
    max_mention_length = model_data['max_mention_length']

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    orig_entity_vocab = EntityVocab(os.path.join(model_dir, 'entity_vocab.tsv'))

    logger.info('Loading dataset...')

    dataset = EntityDisambiguationDataset(data_dir, wikipedia_titles_file, wikipedia_redirects_file)
    entity_titles = []
    for data in dataset.get_all_datasets():
        for document in data:
            for mention in document.mentions:
                entity_titles.append(mention.title)
                for candidate in mention.candidates:
                    entity_titles.append(candidate.title)
    entity_titles = frozenset(entity_titles)

    orig_entity_emb = state_dict['entity_embeddings.entity_embeddings.weight']
    entity_emb = orig_entity_emb.new_zeros((len(entity_titles) + 2, config.hidden_size))
    orig_entity_bias = state_dict['entity_predictions.bias']
    entity_bias = orig_entity_bias.new_zeros(len(entity_titles) + 2)
    entity_vocab = {PAD_TOKEN: 0, MASK_TOKEN: 1}
    entity_emb[1] = orig_entity_emb[orig_entity_vocab[MASK_TOKEN]]
    entity_bias[1] = orig_entity_bias[orig_entity_vocab[MASK_TOKEN]]
    for n, title in enumerate(entity_titles, 2):
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

    def evaluate(model, dataset_name, document_split_mode):
        model.eval()

        eval_documents = getattr(dataset, dataset_name)
        eval_data = generate_data(eval_documents, tokenizer, entity_vocab, 'eval', document_split_mode, max_seq_length,
                                  max_candidate_length, max_mention_length)
        precision, recall, f1 = compute_precision_recall_f1(model, eval_data, entity_vocab, dataset_name,
                                                            min_context_entity_prob)

        tqdm.write(f'***** Eval results: {dataset_name} *****')
        tqdm.write(f'  F1 = {f1:.3f}')
        tqdm.write(f'  Precision = {precision:.3f}')
        tqdm.write(f'  Recall = {recall:.3f}')

        experiment.log_metric(f'{dataset_name}_precision', precision)
        experiment.log_metric(f'{dataset_name}_recall', recall)
        experiment.log_metric(f'{dataset_name}_f1', f1)

        return precision, recall, f1

    results = []

    if in_domain:
        logger.info('Fix entity embeddings during training: %s', fix_entity_emb)
        if fix_entity_emb:
            model.entity_embeddings.entity_embeddings.weight.requires_grad = False

        logger.info('Fix entity bias during training: %s', fix_entity_bias)
        if fix_entity_bias:
            model.entity_predictions.bias.requires_grad = False

        train_data = generate_data(dataset.train, tokenizer, entity_vocab, 'train', 'simple', max_seq_length,
                                   max_candidate_length, max_mention_length)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        grad_avg_device = torch.device('cuda')
        if grad_avg_on_cpu:
            grad_avg_device = torch.device('cpu')
        optimizer = LukeDenseSparseAdam(optimizer_parameters, lr=learning_rate, grad_avg_device=grad_avg_device)
        scheduler = WarmupInverseSquareRootSchedule(optimizer, warmup_steps=warmup_steps)

        best_val_f1_score = 0.0
        best_weights = None
        epoch = 0
        global_step = 0
        num_epochs_without_improvement = 0

        while True:
            with experiment.train():
                model.train()

                mask_id = entity_vocab[MASK_TOKEN]
                train_losses = []

                batches = np.array_split(np.random.permutation(train_data),
                                         math.ceil(len(train_data) / num_documents_per_batch))
                with tqdm(batches) as pbar:
                    for batch in pbar:
                        batch_entity_length = sum(len(item['mentions']) for item in batch)
                        for item in batch:
                            args = {k: torch.as_tensor(v, device='cuda') for k, v in item['features'].items()}
                            entity_ids = args.pop('entity_ids')
                            entity_attention_mask = args.pop('entity_attention_mask')
                            entity_labels = args.pop('entity_labels')
                            entity_length = entity_ids.size()[1]
                            for _ in range(entity_length):
                                logits = model(entity_ids=entity_ids, entity_attention_mask=entity_attention_mask,
                                               **args)
                                probs = F.softmax(logits, dim=2) * (entity_ids == mask_id).unsqueeze(-1).type_as(logits)
                                max_probs, max_indices = torch.max(probs.squeeze(0), dim=1)
                                max_prob, target_index = torch.max(max_probs, dim=0)
                                loss = F.cross_entropy(logits[:, target_index], entity_labels[:, target_index])
                                loss = loss / batch_entity_length
                                loss.backward()
                                train_losses.append(loss.item())

                                entity_ids[0, target_index] = max_indices[target_index]
                                if max_prob <= min_context_entity_prob:
                                    entity_attention_mask[0, target_index] = 0

                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                        pbar.set_description(f'epoch: {epoch} '
                                             f'lr: {scheduler.get_lr()[0]:.8f} '
                                             f'loss: {np.mean(train_losses):.8f}')
                        experiment.log_metric('batch_loss', np.mean(train_losses), step=global_step)
                        experiment.log_metric('learning_rate', scheduler.get_lr()[0], step=global_step)
                        global_step += 1
                        train_losses = []

            epoch += 1

            with experiment.validate():
                val_f1_score = evaluate(model, 'test_a', 'simple')[-1]
                if val_f1_score > best_val_f1_score:
                    best_val_f1_score = val_f1_score
                    best_weights = {k: v.to('cpu').clone() for k, v in model.state_dict().items()}
                    num_epochs_without_improvement = 0
                else:
                    num_epochs_without_improvement += 1

                if num_epochs_without_improvement >= patience:
                    model.load_state_dict(best_weights)
                    break

        with experiment.test():
            results.append(evaluate(model, 'test_b', document_split_mode))

    else:
        if not test_set:
            test_set = ['ace2004', 'aquaint', 'msnbc', 'wikipedia', 'clueweb']

        with experiment.test():
            for dataset_name in test_set:
                results.append(evaluate(model, dataset_name, document_split_mode))

    return results


def compute_precision_recall_f1(model, eval_data, entity_vocab, desc, min_context_entity_prob):
    mask_id = entity_vocab[MASK_TOKEN]

    predictions = []
    labels = []
    candidate_flags = []
    for item in tqdm(eval_data, desc=desc, leave=False):
        args = {k: torch.as_tensor(v, device='cuda') for k, v in item['features'].items()}
        entity_ids = args.pop('entity_ids')
        entity_attention_mask = args.pop('entity_attention_mask')
        entity_labels = args.pop('entity_labels')
        entity_length = entity_ids.size()[1]
        with torch.no_grad():
            for _ in range(entity_length):
                logits = model(entity_ids=entity_ids, entity_attention_mask=entity_attention_mask, **args)
                probs = F.softmax(logits, dim=2) * (entity_ids == mask_id).unsqueeze(-1).type_as(logits)
                max_probs, max_indices = torch.max(probs.squeeze(0), dim=1)
                max_prob, target_index = torch.max(max_probs, dim=0)

                entity_ids[0, target_index] = max_indices[target_index]
                if max_prob <= min_context_entity_prob:
                    entity_attention_mask[0, target_index] = 0

        for target_index in item['target_mention_indices']:
            predictions.append(entity_ids[0, target_index].item())
            labels.append(entity_labels[0, target_index].item())
            candidate_flags.append(bool(item['mentions'][target_index].candidates))

    num_correct = 0
    num_mentions = 0
    num_mentions_with_candidates = 0
    for prediction, label, candidate_flag in zip(predictions, labels, candidate_flags):
        if prediction == label:
            num_correct += 1

        assert not (candidate_flag == 1 and prediction == 0)
        assert label != 0

        num_mentions += 1
        if candidate_flag:
            num_mentions_with_candidates += 1

    logger.info('#mentions (%s): %d', desc, num_mentions)
    logger.info('#mentions with candidates (%s): %d', desc, num_mentions_with_candidates)
    logger.info('#correct (%s): %d', desc, num_correct)

    precision = num_correct / num_mentions_with_candidates
    recall = num_correct / num_mentions
    f1 = 2.0 * precision * recall / (precision + recall)

    return precision, recall, f1


if __name__ == '__main__':
    run()
