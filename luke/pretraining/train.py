import contextlib
import inspect
import json
import logging
import math
import os
import subprocess
import time
import click
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from pytorch_transformers.modeling_bert import BertConfig, BertForPreTraining
from pytorch_transformers.optimization import ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule
from tqdm import tqdm

from luke.model import LukeConfig, LukeE2EConfig
from luke.pretraining.batch_generator import LukePretrainingBatchGenerator
from luke.pretraining.dataset import WikipediaPretrainingDataset, ENTITY_LINKER_FILE, ENTITY_VOCAB_FILE
from luke.pretraining.model import LukePretrainingModel, LukeE2EPretrainingModel
from luke.optimization import LukeDenseSparseAdam

logger = logging.getLogger(__name__)

NCCL_SOCKET_IF_NAME = 'lo'
MASTER_ADDR = '127.0.0.1'
MASTER_PORT = '29502'


def run_pretraining(dataset_dir, output_dir, parallel, mode, bert_model_name, batch_size, gradient_accumulation_steps,
                    learning_rate, lr_schedule, warmup_steps, adam_b1, adam_b2, max_grad_norm, masked_lm_prob,
                    masked_entity_prob, whole_word_masking, fix_bert_weights, grad_avg_on_cpu, num_epochs, fp16,
                    fp16_opt_level, local_rank, log_dir, model_file, optimizer_file, scheduler_file, save_interval_sec,
                    num_el_hidden_layers=None, entity_selector_softmax_temp=None, global_step=0):
    train_args = {}
    for arg in inspect.getfullargspec(run_pretraining).args:
        train_args[arg] = locals()[arg]

    if parallel and local_rank == -1:
        run_parallel_pretraining(**train_args)
        return

    if local_rank == -1:
        device = torch.device('cuda')
        num_workers = 1
        worker_index = 0
    else:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device('cuda', local_rank)
        num_workers = torch.distributed.get_world_size()
        worker_index = torch.distributed.get_rank()

    dataset = WikipediaPretrainingDataset(dataset_dir)
    bert_config = BertConfig.from_pretrained(bert_model_name)

    num_train_steps_per_epoch = math.ceil(len(dataset) / batch_size)
    num_train_steps = math.ceil(len(dataset) / batch_size * num_epochs)
    logger.info('The number of training steps: %d', num_train_steps)
    train_batch_size = int(batch_size / gradient_accumulation_steps / num_workers)

    if mode == 'default':
        config = LukeConfig(entity_vocab_size=dataset.entity_vocab.size, **bert_config.to_dict())
        model = LukePretrainingModel(config)
        batch_generator = LukePretrainingBatchGenerator(dataset_dir, mode, train_batch_size, masked_lm_prob,
                                                        masked_entity_prob, whole_word_masking, num_workers=num_workers,
                                                        worker_index=worker_index, skip=global_step * batch_size)
    elif mode == 'e2e':
        config = LukeE2EConfig(entity_vocab_size=dataset.entity_vocab.size,
                               num_el_hidden_layers=num_el_hidden_layers,
                               entity_selector_softmax_temp=entity_selector_softmax_temp,
                               **bert_config.to_dict())
        model = LukeE2EPretrainingModel(config)
        batch_generator = LukePretrainingBatchGenerator(dataset_dir, mode, train_batch_size, masked_lm_prob,
                                                        masked_entity_prob, whole_word_masking, num_workers=num_workers,
                                                        worker_index=worker_index, skip=global_step * batch_size)
    else:
        raise RuntimeError(f'Invalid mode: {mode}')

    logger.info('Model configuration: %s', config)

    if model_file is None:
        bert_model = BertForPreTraining.from_pretrained(bert_model_name)
        bert_state_dict = bert_model.state_dict()
        if mode == 'e2e':
            for key in tuple(bert_state_dict.keys()):
                if key.startswith('bert.encoder.layer.') and int(key.split('.')[3]) < num_el_hidden_layers:
                    bert_state_dict['el_encoder.' + key[13:]] = bert_state_dict[key]
        model.load_bert_weights(bert_state_dict)
    else:
        model_state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    if fix_bert_weights:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.entity_embeddings.parameters():
            param.requires_grad = True
        for param in model.entity_predictions.parameters():
            param.requires_grad = True
        if mode == 'e2e':
            for param in model.entity_selector.parameters():
                param.requires_grad = True
            for param in model.el_encoder.parameters():
                param.requires_grad = True

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    grad_avg_device = device
    if grad_avg_on_cpu:
        grad_avg_device = torch.device('cpu')
    optimizer = LukeDenseSparseAdam(optimizer_parameters, lr=learning_rate, betas=(adam_b1, adam_b2),
                                    max_grad_norm=max_grad_norm, grad_avg_device=grad_avg_device)

    if fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if optimizer_file is not None:
        optimizer.load_state_dict(torch.load(optimizer_file, map_location='cpu'))

    if lr_schedule == 'warmup_constant':
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)
    elif lr_schedule == 'warmup_linear':
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_steps)
    else:
        scheduler = ConstantLRSchedule(optimizer)

    if scheduler_file is not None:
        scheduler.load_state_dict(torch.load(scheduler_file, map_location='cpu'))

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          broadcast_buffers=False, find_unused_parameters=True)

    model.train()

    dataset.tokenizer.save_pretrained(output_dir)
    dataset.entity_vocab.save(os.path.join(output_dir, ENTITY_VOCAB_FILE))
    dataset.entity_linker.save(os.path.join(output_dir, ENTITY_LINKER_FILE))
    metadata = dict(model_config=config.to_dict(),
                    max_seq_length=dataset.max_seq_length,
                    max_entity_length=dataset.max_entity_length,
                    max_mention_length=dataset.max_mention_length,
                    arguments=train_args)
    if mode == 'e2e':
        metadata['max_candidate_length'] = dataset.max_candidate_length
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    def save_model(model, suffix):
        if local_rank != -1:
            model = model.module

        model_file = f'model_{suffix}.bin'
        torch.save(model.state_dict(), os.path.join(output_dir, model_file))
        optimizer_file = f'optimizer_{suffix}.bin'
        torch.save(optimizer.state_dict(), os.path.join(output_dir, optimizer_file))
        scheduler_file = f'scheduler_{suffix}.bin'
        torch.save(scheduler.state_dict(), os.path.join(output_dir, scheduler_file))
        metadata = dict(global_step=global_step,
                        model_file=model_file,
                        optimizer_file=optimizer_file,
                        scheduler_file=scheduler_file)
        with open(os.path.join(output_dir, f'metadata_{suffix}.json'), 'w') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    if local_rank in (0, -1):
        summary_writer = SummaryWriter(log_dir)
        pbar = tqdm(total=num_train_steps, initial=global_step)

    tr_loss = 0
    accumulation_count = 0
    results = []
    prev_error = False
    prev_step_time = time.time()
    prev_save_time = time.time()

    for batch in batch_generator.generate_batches():
        try:
            batch = {k: torch.from_numpy(v).to(device) for k, v in batch.items()}
            result = model(**batch)
            loss = result['loss']
            result = {k: v.to('cpu').detach().numpy() for k, v in result.items()}

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            def maybe_no_sync():
                if hasattr(model, 'no_sync') and num_workers > 1 and accumulation_count + 1 != gradient_accumulation_steps:
                    return model.no_sync()
                else:
                    return contextlib.ExitStack()

            with maybe_no_sync():
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

        except RuntimeError:
            if prev_error:
                logger.exception('Consecutive errors have been observed. Exiting...')
                raise
            logger.exception('An unexpected error has occurred. Skipping a batch...')
            prev_error = True
            loss = None
            torch.cuda.empty_cache()
            continue

        accumulation_count += 1
        prev_error = False
        tr_loss += loss.item()
        loss = None
        results.append(result)

        if accumulation_count == gradient_accumulation_steps:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            accumulation_count = 0

            summary = {}
            summary['learning_rate'] = max(scheduler.get_lr())
            summary['loss'] = tr_loss
            tr_loss = 0

            current_time = time.time()
            summary['batch_run_time'] = current_time - prev_step_time
            prev_step_time = current_time

            for name in ('masked_lm', 'masked_entity', 'entity_selector'):
                try:
                    summary[name + '_loss'] = np.concatenate([r[name + '_loss'].flatten() for r in results]).mean()
                    correct = np.concatenate([r[name + '_correct'].flatten() for r in results]).sum()
                    total = np.concatenate([r[name + '_total'].flatten() for r in results]).sum()
                    if total > 0:
                        summary[name + '_acc'] = correct / total
                except KeyError:
                    continue

            results = []

            if local_rank in (0, -1):
                for (name, value) in summary.items():
                    summary_writer.add_scalar(name, value, global_step)
                desc = f'epoch: {int(global_step / num_train_steps_per_epoch)} loss: {summary["loss"]:.4f}'
                pbar.set_description(desc)
                pbar.update()

            global_step += 1

            if local_rank in (0, -1):
                if global_step % num_train_steps_per_epoch == 0:
                    epoch = global_step / num_train_steps_per_epoch
                    save_model(model, f'epoch{epoch}')
                if save_interval_sec and time.time() - prev_save_time > save_interval_sec:
                    save_model(model, f'step{global_step:07}')
                    prev_save_time = time.time()

            if global_step == num_train_steps:
                break

    if local_rank in (0, -1):
        summary_writer.close()


def run_parallel_pretraining(**kwargs):
    num_workers = torch.cuda.device_count()
    current_env = os.environ.copy()
    current_env['NCCL_SOCKET_IFNAME'] = NCCL_SOCKET_IF_NAME
    current_env['MASTER_ADDR'] = MASTER_ADDR
    current_env['MASTER_PORT'] = MASTER_PORT
    current_env['WORLD_SIZE'] = str(num_workers)
    current_env['OMP_NUM_THREADS'] = str(1)
    processes = []
    for local_rank in range(num_workers):
        cmd = ['luke', 'start-pretraining-worker', f'--local-rank={local_rank}', f'--kwargs={json.dumps(kwargs)}']
        current_env['RANK'] = str(local_rank)
        current_env['LOCAL_RANK'] = str(local_rank)
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    try:
        for process in processes:
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()


from luke.cli import cli

@cli.command()
@click.argument('dataset_dir', type=click.Path(file_okay=False, exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--mode', type=click.Choice(['default', 'e2e']), default='default')
@click.option('--parallel', is_flag=True)
@click.option('--bert-model-name', default='bert-base-uncased')
@click.option('--batch-size', default=256)
@click.option('--gradient-accumulation-steps', default=1)
@click.option('--learning-rate', default=1e-4)
@click.option('--lr-schedule', type=click.Choice(['none', 'warmup_constant', 'warmup_linear']), default='warmup_linear')
@click.option('--warmup-steps', default=0)
@click.option('--adam-b1', default=0.9)
@click.option('--adam-b2', default=0.999)
@click.option('--max-grad-norm', default=1.0)
@click.option('--masked-lm-prob', default=0.15)
@click.option('--masked-entity-prob', default=0.3)
@click.option('--whole-word-masking', is_flag=True)
@click.option('--num-el-hidden-layers', default=3)
@click.option('--entity-selector-softmax-temp', default=0.1)
@click.option('--fix-bert-weights', is_flag=True)
@click.option('--grad-avg-on-cpu', is_flag=True)
@click.option('--num-epochs', default=5)
@click.option('--fp16', is_flag=True)
@click.option('--fp16-opt-level', default='O1', type=click.Choice(['O0', 'O1', 'O2', 'O3']))
@click.option('--local-rank', '--local_rank', default=-1)
@click.option('--log-dir', type=click.Path(), default=None)
@click.option('--model-file', type=click.Path(exists=True), default=None)
@click.option('--optimizer-file', type=click.Path(exists=True), default=None)
@click.option('--scheduler-file', type=click.Path(exists=True), default=None)
@click.option('--save-interval-sec', default=1800)
def pretrain(**kwargs):
    run_pretraining(**kwargs)


@cli.command()
@click.argument('output_dir', type=click.Path())
@click.option('--batch-size', default=None, type=int)
@click.option('--gradient-accumulation-steps', default=None, type=int)
def resume_pretraining(output_dir, **kwargs):
    with open(os.path.join(output_dir, 'metadata.json')) as f:
        args = json.load(f)['arguments']

    step_metadata_file = sorted([f for f in os.listdir(output_dir)
                                 if f.startswith('metadata_') and f.endswith('.json')])[-1]
    with open(os.path.join(output_dir, step_metadata_file)) as f:
        step_metadata = json.load(f)

    args['model_file'] = os.path.join(output_dir, step_metadata['model_file'])
    args['optimizer_file'] = os.path.join(output_dir, step_metadata['optimizer_file'])
    args['scheduler_file'] = os.path.join(output_dir, step_metadata['scheduler_file'])
    args['global_step'] = step_metadata['global_step']

    for key, value in kwargs.items():
        if value is not None:
            args[key] = value

    run_pretraining(**args)


@cli.command(hidden=True)
@click.option('--local-rank', type=int)
@click.option('--kwargs', default='{}')
def start_pretraining_worker(local_rank, kwargs):
    kwargs = json.loads(kwargs)
    kwargs['local_rank'] = local_rank
    run_pretraining(**kwargs)
