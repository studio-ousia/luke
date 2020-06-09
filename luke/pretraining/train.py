import contextlib
import datetime
import json
import logging
import math
import os
import subprocess
import time
from argparse import Namespace

import click
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from luke.model import LukeConfig
from luke.optimization import LukeDenseSparseAdam
from luke.pretraining.batch_generator import LukePretrainingBatchGenerator, MultilingualBatchGenerator
from luke.pretraining.dataset import (
    WikipediaPretrainingDataset,
    MultilingualPretrainingDataset,
    ENTITY_VOCAB_FILE,
    MULTILINGUAL_ENTITY_VOCAB_FILE,
)
from luke.pretraining.model import LukePretrainingModel

from luke.utils.model_utils import get_pretrained_model, get_config

logger = logging.getLogger(__name__)


@click.command()
@click.argument("dataset_dir")
@click.argument("output_dir", type=click.Path())
@click.option("--multilingual", is_flag=True)
@click.option("--sampling-smoothing", default=0.7)
@click.option("--parallel", is_flag=True)
@click.option("--cpu", is_flag=True)
@click.option("--bert-model-name", default="bert-base-uncased")
@click.option("--entity-emb-size", default=None, type=int)
@click.option("--batch-size", default=256)
@click.option("--gradient-accumulation-steps", default=1)
@click.option("--learning-rate", default=1e-4)
@click.option("--lr-schedule", type=click.Choice(["warmup_constant", "warmup_linear"]), default="warmup_linear")
@click.option("--warmup-steps", default=1000)
@click.option("--adam-b1", default=0.9)
@click.option("--adam-b2", default=0.999)
@click.option("--adam-eps", default=1e-6)
@click.option("--weight-decay", default=0.01)
@click.option("--max-grad-norm", default=0.0)
@click.option("--masked-lm-prob", default=0.15)
@click.option("--masked-entity-prob", default=0.15)
@click.option("--whole-word-masking", is_flag=True)
@click.option("--fix-bert-weights", is_flag=True)
@click.option("--grad-avg-on-cpu/--grad-avg-on-gpu", default=False)
@click.option("--num-epochs", default=5)
@click.option("--global-step", default=0)
@click.option("--fp16", is_flag=True)
@click.option("--fp16-opt-level", default="O2", type=click.Choice(["O1", "O2"]))
@click.option("--fp16-master-weights/--fp16-no-master-weights", default=True)
@click.option("--fp16-min-loss-scale", default=1)
@click.option("--fp16-max-loss-scale", default=4)
@click.option("--local-rank", "--local_rank", default=-1)
@click.option("--num-nodes", default=1)
@click.option("--node-rank", default=0)
@click.option("--master-addr", default="127.0.0.1")
@click.option("--master-port", default="29502")
@click.option("--log-dir", type=click.Path(), default=None)
@click.option("--model-file", type=click.Path(exists=True), default=None)
@click.option("--optimizer-file", type=click.Path(exists=True), default=None)
@click.option("--scheduler-file", type=click.Path(exists=True), default=None)
@click.option("--amp-file", type=click.Path(exists=True), default=None)
@click.option("--save-interval-sec", default=None, type=int)
@click.option("--save-interval-steps", default=None, type=int)
def pretrain(**kwargs):
    run_pretraining(Namespace(**kwargs))


@click.command()
@click.argument("output_dir", type=click.Path())
@click.option("--batch-size", default=None, type=int)
@click.option("--gradient-accumulation-steps", default=None, type=int)
@click.option("--grad-avg-on-cpu/--grad-avg-on-gpu", default=False)
@click.option("--num-nodes", default=1)
@click.option("--node-rank", default=0)
@click.option("--master-addr", default="127.0.0.1")
@click.option("--master-port", default="29502")
def resume_pretraining(output_dir: str, **kwargs):
    if "num_nodes" not in kwargs:
        kwargs["num_nodes"] = 1
        kwargs["node_rank"] = 0
        kwargs["master_addr"] = "127.0.0.1"
        kwargs["master_port"] = "29502"

    with open(os.path.join(output_dir, "metadata.json")) as f:
        args = json.load(f)["arguments"]

    step_metadata_file = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("metadata_") and f.endswith(".json")]
    )[-1]
    with open(os.path.join(output_dir, step_metadata_file)) as f:
        step_metadata = json.load(f)

    args["model_file"] = os.path.join(output_dir, step_metadata["model_file"])
    args["optimizer_file"] = os.path.join(output_dir, step_metadata["optimizer_file"])
    args["scheduler_file"] = os.path.join(output_dir, step_metadata["scheduler_file"])
    if "amp_file" in step_metadata:
        args["amp_file"] = os.path.join(output_dir, step_metadata["amp_file"])
    else:
        args["amp_file"] = None
    args["global_step"] = step_metadata["global_step"]
    args["local_rank"] = -1

    for key, value in kwargs.items():
        if value is not None:
            args[key] = value

    run_pretraining(Namespace(**args))


@click.command(hidden=True)
@click.option("--local-rank", type=int)
@click.option("--args", default="{}")
def start_pretraining_worker(local_rank: int, args):
    args = json.loads(args)
    args["local_rank"] = local_rank
    run_pretraining(Namespace(**args))


def run_pretraining(args):
    if args.parallel and args.local_rank == -1:
        run_parallel_pretraining(args)
        return

    if args.local_rank == -1:
        if args.cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        num_workers = 1
        worker_index = 0
    else:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
        num_workers = torch.distributed.get_world_size()
        worker_index = torch.distributed.get_rank()

    if args.local_rank not in (-1, 0):
        logging.getLogger().setLevel(logging.WARN)

    logger.info("Starting pretraining with the following arguments: %s", args)

    if args.multilingual:
        dataset_dir_list = args.dataset_dir.split(",")
        dataset = MultilingualPretrainingDataset(dataset_dir_list)
        entity_vocab_file = MULTILINGUAL_ENTITY_VOCAB_FILE
    else:
        dataset = WikipediaPretrainingDataset(args.dataset_dir)
        entity_vocab_file = ENTITY_VOCAB_FILE

    bert_config = get_config(args.bert_model_name)

    num_train_steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    num_train_steps = math.ceil(len(dataset) / args.batch_size * args.num_epochs)
    logger.info("The number of training steps: %d", num_train_steps)
    train_batch_size = int(args.batch_size / args.gradient_accumulation_steps / num_workers)

    if args.entity_emb_size is None:
        args.entity_emb_size = bert_config.hidden_size

    config = LukeConfig(
        entity_vocab_size=dataset.entity_vocab.size,
        bert_model_name=args.bert_model_name,
        entity_emb_size=args.entity_emb_size,
        **bert_config.to_dict(),
    )
    model = LukePretrainingModel(config)

    global_step = args.global_step

    if args.multilingual:
        batch_generator = MultilingualBatchGenerator(
            dataset_dir_list,
            dataset.data_size_list,
            args.sampling_smoothing,
            train_batch_size,
            args.masked_lm_prob,
            args.masked_entity_prob,
            args.whole_word_masking,
            num_workers=num_workers,
            worker_index=worker_index,
            skip=global_step * args.batch_size,
        )

    else:
        batch_generator = LukePretrainingBatchGenerator(
            args.dataset_dir,
            train_batch_size,
            args.masked_lm_prob,
            args.masked_entity_prob,
            args.whole_word_masking,
            num_workers=num_workers,
            worker_index=worker_index,
            skip=global_step * args.batch_size,
        )

    logger.info("Model configuration: %s", config)

    if args.fix_bert_weights:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.entity_embeddings.parameters():
            param.requires_grad = True
        for param in model.entity_predictions.parameters():
            param.requires_grad = True

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    grad_avg_device = device
    if args.grad_avg_on_cpu:
        grad_avg_device = torch.device("cpu")
    optimizer = LukeDenseSparseAdam(
        optimizer_parameters,
        lr=args.learning_rate,
        betas=(args.adam_b1, args.adam_b2),
        eps=args.adam_eps,
        grad_avg_device=grad_avg_device,
    )

    if args.fp16:
        from apex import amp

        if args.fp16_opt_level == "O2":
            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level=args.fp16_opt_level,
                master_weights=args.fp16_master_weights,
                min_loss_scale=args.fp16_min_loss_scale,
                max_loss_scale=args.fp16_max_loss_scale,
            )
        else:
            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level=args.fp16_opt_level,
                min_loss_scale=args.fp16_min_loss_scale,
                max_loss_scale=args.fp16_max_loss_scale,
            )

    if args.model_file is None:
        bert_model = get_pretrained_model(args.bert_model_name)
        bert_state_dict = bert_model.state_dict()
        model.load_bert_weights(bert_state_dict)

    else:
        model_state_dict = torch.load(args.model_file, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)

    if args.optimizer_file is not None:
        optimizer.load_state_dict(torch.load(args.optimizer_file, map_location="cpu"))

    if args.amp_file is not None:
        amp.load_state_dict(torch.load(args.amp_file, map_location="cpu"))

    if args.lr_schedule == "warmup_constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == "warmup_linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_steps
        )
    else:
        raise RuntimeError(f"Invalid scheduler: {args.lr_schedule}")

    if args.scheduler_file is not None:
        scheduler.load_state_dict(torch.load(args.scheduler_file, map_location="cpu"))

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    model.train()

    # if args.local_rank in (0, -1):
    if args.local_rank == -1 or worker_index == 0:
        dataset.tokenizer.save_pretrained(args.output_dir)

        dataset.entity_vocab.save(os.path.join(args.output_dir, entity_vocab_file))
        metadata = dict(
            model_config=config.to_dict(),
            max_seq_length=dataset.max_seq_length,
            max_entity_length=dataset.max_entity_length,
            max_mention_length=dataset.max_mention_length,
            arguments=vars(args),
        )
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    def save_model(model, suffix):
        if args.local_rank != -1:
            model = model.module

        model_file = f"model_{suffix}.bin"
        torch.save(model.state_dict(), os.path.join(args.output_dir, model_file))
        optimizer_file = f"optimizer_{suffix}.bin"
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, optimizer_file))
        scheduler_file = f"scheduler_{suffix}.bin"
        torch.save(scheduler.state_dict(), os.path.join(args.output_dir, scheduler_file))
        metadata = dict(
            global_step=global_step, model_file=model_file, optimizer_file=optimizer_file, scheduler_file=scheduler_file
        )
        if args.fp16:
            amp_file = f"amp_{suffix}.bin"
            torch.save(amp.state_dict(), os.path.join(args.output_dir, amp_file))
            metadata["amp_file"] = amp_file
        with open(os.path.join(args.output_dir, f"metadata_{suffix}.json"), "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    # if args.local_rank in (0, -1):
    if args.local_rank == -1 or worker_index == 0:
        summary_writer = SummaryWriter(args.log_dir)
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
            loss = result["loss"]
            result = {k: v.to("cpu").detach().numpy() for k, v in result.items()}

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            def maybe_no_sync():
                if (
                    hasattr(model, "no_sync")
                    and num_workers > 1
                    and accumulation_count + 1 != args.gradient_accumulation_steps
                ):
                    return model.no_sync()
                else:
                    return contextlib.ExitStack()

            with maybe_no_sync():
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

        except RuntimeError:
            if prev_error:
                logger.exception("Consecutive errors have been observed. Exiting...")
                raise
            logger.exception("An unexpected error has occurred. Skipping a batch...")
            prev_error = True
            loss = None
            torch.cuda.empty_cache()
            continue

        accumulation_count += 1
        prev_error = False
        tr_loss += loss.item()
        loss = None
        results.append(result)

        if accumulation_count == args.gradient_accumulation_steps:
            if args.max_grad_norm != 0.0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            accumulation_count = 0

            summary = {}
            summary["learning_rate"] = max(scheduler.get_lr())
            summary["loss"] = tr_loss
            tr_loss = 0

            current_time = time.time()
            summary["batch_run_time"] = current_time - prev_step_time
            prev_step_time = current_time

            for name in ("masked_lm", "masked_entity"):
                try:
                    summary[name + "_loss"] = np.concatenate([r[name + "_loss"].flatten() for r in results]).mean()
                    correct = np.concatenate([r[name + "_correct"].flatten() for r in results]).sum()
                    total = np.concatenate([r[name + "_total"].flatten() for r in results]).sum()
                    if total > 0:
                        summary[name + "_acc"] = correct / total
                except KeyError:
                    continue

            results = []

            if args.local_rank == -1 or worker_index == 0:
                # if args.local_rank in (0, -1):
                for (name, value) in summary.items():
                    summary_writer.add_scalar(name, value, global_step)
                desc = (
                    f"epoch: {int(global_step / num_train_steps_per_epoch)} "
                    f'loss: {summary["loss"]:.4f} '
                    f'time: {datetime.datetime.now().strftime("%H:%M:%S")}'
                )
                pbar.set_description(desc)
                pbar.update()

            global_step += 1

            if args.local_rank == -1 or worker_index == 0:
                # if args.local_rank in (0, -1):
                if global_step == num_train_steps:
                    # save the final model
                    save_model(model, f"epoch{args.num_epochs}")
                    time.sleep(60)
                elif global_step % num_train_steps_per_epoch == 0:
                    # save the model at each epoch
                    epoch = int(global_step / num_train_steps_per_epoch)
                    save_model(model, f"epoch{epoch}")
                if args.save_interval_sec and time.time() - prev_save_time > args.save_interval_sec:
                    save_model(model, f"step{global_step:07}")
                    prev_save_time = time.time()
                if args.save_interval_steps and global_step % args.save_interval_steps == 0:
                    save_model(model, f"step{global_step}")

            # if global_step == num_train_steps:
            #     break

    # if args.local_rank in (0, -1):
    if args.local_rank == -1 or worker_index == 0:
        summary_writer.close()


def run_parallel_pretraining(args):
    num_workers = torch.cuda.device_count()

    current_env = os.environ.copy()
    # current_env['NCCL_SOCKET_IFNAME'] = NCCL_SOCKET_IF_NAME
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = args.master_port
    # current_env['WORLD_SIZE'] = str(num_workers)
    current_env["WORLD_SIZE"] = str(num_workers * args.num_nodes)
    current_env["OMP_NUM_THREADS"] = str(1)
    processes = []
    for local_rank in range(num_workers):
        cmd = ["luke", "start-pretraining-worker", f"--local-rank={local_rank}", f"--args={json.dumps(vars(args))}"]
        current_env["RANK"] = str(num_workers * args.node_rank + local_rank)
        # current_env['RANK'] = str(local_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
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
