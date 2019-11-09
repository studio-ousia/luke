import datetime
import json
import logging
import os
import string
import random
import subprocess
import tempfile
import time
import click

import main

logger = logging.getLogger(__name__)

BASE_SCRIPT = """#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N {job_name}
#$ -e {stderr_file}
#$ -o {stdout_file}
#$ -jc {node_type}-container_g{num_gpus}.{hour}h
#$ -ac d=nvcr-pytorch-1901
{headers}

. /fefs/opt/dgx/env_set/common_env_set.sh
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/lib:$HOME/local/lib

source .venv/bin/activate
{cmd}
"""


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option('--run-name-suffix', type=click.Path(),
              default=datetime.datetime.now().strftime('_%Y%m%d-%H%M%S'))
# @click.option('--base-log-dir', type=click.Path(), default='log_glue')
@click.option('--base-output-dir', type=click.Path(), default='out_glue')
@click.option('--num-gpus', default=1)
@click.option('--node-type', default='gpuhss')
@click.option('--hour', default='24', type=click.Choice(['24', '72', '168']))
def run(run_name_suffix, base_output_dir, num_gpus, node_type, hour, **kwargs):
    output_dir = os.path.join(base_output_dir, kwargs['task_name'] + run_name_suffix)
    # kwargs['output_dir'] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # log_dir = os.path.join(base_log_dir, run_name)
    # kwargs['log_dir'] = log_dir
    # os.makedirs(log_dir, exist_ok=True)

    suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    stdout_file = os.path.join(output_dir, 'stdout_%s.txt' % suffix)
    stderr_file = os.path.join(output_dir, 'stderr_%s.txt' % suffix)

    job_name = kwargs['task_name'] + run_name_suffix + '_' + random_string(6)
    json_data = json.dumps(kwargs)
    cmd = f"python scripts/glue/raiden_cli.py run-json main.run.callback --json-data='{json_data}'"
    submit_command(cmd, job_name, num_gpus, node_type, hour, stdout_file, stderr_file)

    while True:
        if os.path.exists(stderr_file):
            subprocess.run(['tail', '-f', stderr_file], check=True)
            break
        time.sleep(5)

run.params =\
    [p for p in main.run.params if p.name not in ('output_dir', 'log_dir')] + run.params


@cli.command()
@click.argument('func_name')
@click.option('-j', '--json-data', default=None)
def run_json(func_name, json_data, **kwargs):
    kwargs = json.loads(json_data)
    logger.info('Arguments: %s', json.dumps(kwargs, indent=2, sort_keys=True))
    func = eval(func_name)
    func(**kwargs)


def submit_command(cmd, job_name, num_gpus, node_type, hour, stdout_file, stderr_file, headers=''):
    script = BASE_SCRIPT.format(**locals())

    click.echo('')
    click.echo('Job script:')
    click.echo(script)
    click.echo('---')
    with tempfile.NamedTemporaryFile('w', prefix='marco_train_') as f:
        f.write(script)
        f.flush()
        subprocess.run(['qsub', f.name], check=True)


def random_string(length):
    return ''.join(random.choice(string.ascii_letters) for m in range(length))


if __name__ == '__main__':
    cli()

