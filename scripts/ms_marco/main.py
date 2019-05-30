import gc
import inspect
import json
import logging
import os
import click
import joblib
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from luke.optimization import BertAdam

from batch_generator import BatchGenerator
from ranking_model import LukeForReranking, LukeForRerankingConfig

logger = logging.getLogger(__name__)


@click.command()
@click.argument('model_file', type=click.Path())
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--data-dir', type=click.Path(exists=True), default='ms_marco_data')
@click.option('--log-dir', type=click.Path(exists=True), default=None)
@click.option('-v', '--verbose', is_flag=True)
@click.option('--max-seq-length', default=512)
@click.option('--batch-size', default=32)
@click.option('--learning-rate', default=1e-6)
@click.option('--eval-batch-size', default=8)
@click.option('--lr-decay/--no-lr-decay', default=False)
@click.option('--gradient-accumulation-steps', default=1)
@click.option('--num-train-steps', default=400000)
@click.option('--num-warmup-steps', default=40000)
@click.option('--max-entity-length', default=128)
@click.option('--max-mention-length', default=30)
@click.option('--fix-entity-emb/--update-entity-emb', default=True)
@click.option('--use-entities/--no-entities', default=True)
@click.option('--save-every', default=1000)
@click.option('--allocate-gpu-for-optimizer', is_flag=True)
@click.option('--scalar-mix', is_flag=True)
def run(model_file, output_dir, data_dir, verbose, max_seq_length, max_entity_length,
        max_mention_length, batch_size, eval_batch_size, learning_rate, lr_decay,
        gradient_accumulation_steps, num_train_steps, num_warmup_steps, fix_entity_emb,
        use_entities, save_every, allocate_gpu_for_optimizer, scalar_mix=False, log_dir=None,
        optimizer_file=None, global_step=0, num_processed=0):
    run_args = {}
    for arg in inspect.getfullargspec(run.callback).args:
        run_args[arg] = locals()[arg]

    log_format = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    logger.info('Loading model and configurations...')

    state_dict = torch.load(model_file + '.bin', map_location='cpu')

    json_file = model_file + '.json'
    with open(json_file) as f:
        model_data = json.load(f)

    model_config = model_data['model_config']
    model_config['scalar_mix'] = scalar_mix

    config = LukeForRerankingConfig(**model_config)
    logger.info('Model configuration: %s', config)

    model = LukeForReranking(config)

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

    device = torch.device('cuda:0')
    n_gpu = torch.cuda.device_count()

    model.to(device)
    if n_gpu > 1:
        if allocate_gpu_for_optimizer:
            model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu - 1)))
        else:
            model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

    train_batch_size = int(batch_size / gradient_accumulation_steps)
    train_data_file = os.path.join(data_dir, 'dataset_train.tf')

    batch_generator = BatchGenerator(train_data_file, train_batch_size, max_seq_length,
                                     max_entity_length, max_mention_length, use_entities)

    warmup_proportion = num_warmup_steps / num_train_steps

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    if allocate_gpu_for_optimizer:
        optimizer_device = torch.device('cuda:' + str(n_gpu - 1))
    else:
        optimizer_device = torch.device('cuda:0')
    optimizer = BertAdam(optimizer_parameters, lr=learning_rate, lr_decay=lr_decay,
                         device=optimizer_device, warmup=warmup_proportion, t_total=num_train_steps)
    if optimizer_file is not None:
        optimizer.load_state_dict(torch.load(optimizer_file, map_location='cpu'))

    pbar = tqdm(total=num_train_steps, initial=global_step)

    model.train()

    def save_model(model, suffix, global_step, num_processed):
        if n_gpu > 1:
            torch.save(model.module.state_dict(), os.path.join(output_dir, 'model_%s.bin' % suffix))
            config_dict = model.module.config.to_dict()
        else:
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_%s.bin' % suffix))
            config_dict = model.config.to_dict()

        json_data = dict(model_config=config_dict, global_step=global_step)
        with open(os.path.join(output_dir, 'model_%s.json' % suffix), 'w') as f:
            json.dump(json_data, f, indent=2, sort_keys=True)

        model_data = {}
        model_data['args'] = run_args
        model_data['global_step'] = global_step
        model_data['num_processed'] = num_processed
        model_data['model_config'] = config_dict
        joblib.dump(model_data, os.path.join(output_dir, 'model_%s.pkl' % suffix))

        optimizer_file = 'optimizer_%s.bin' % suffix
        torch.save(optimizer.state_dict(), os.path.join(output_dir, optimizer_file))

    summary_writer = SummaryWriter(log_dir)

    step = 0
    tr_loss = 0
    results = []
    for batch in batch_generator.generate_batches(num_skip=num_processed):
        batch = {k: torch.from_numpy(v).to(device) for (k, v) in batch.items()}

        try:
            result = model(**batch)
            loss = result['loss']
            result = {k: v.cpu().detach().numpy() for (k, v) in result.items()}
            if n_gpu > 1:
                loss = loss.mean()

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.exception('Out of memory error has occurred. Skipping a batch...')
                loss = None
                gc.collect()
                torch.cuda.empty_cache()
                continue
            else:
                raise

        step += 1
        tr_loss += loss.item()
        num_processed += batch['label'].size()[0]
        results.append(result)
        loss = None

        if step == gradient_accumulation_steps:
            optimizer.step()
            model.zero_grad()

            summary_writer.add_scalar('learning_rate', max(optimizer.get_lr()), global_step)
            summary_writer.add_scalar('loss', tr_loss, global_step)
            correct = np.concatenate([r['correct'].flatten() for r in results]).sum()
            total = np.concatenate([r['total'].flatten() for r in results]).sum()
            if total > 0:
                summary_writer.add_scalar('accuracy', float(correct) / total, global_step)

            step = 0
            global_step += 1
            tr_loss = 0
            results = []

            if global_step % save_every == 0:
                save_model(model, 'step%04d' % (global_step,), global_step, num_processed)

            pbar.update(1)

    summary_writer.close()


if __name__ == '__main__':
    run()
