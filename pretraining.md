# Pretraining LUKE

This document describes how to pretrain LUKE on your machine.

## 1. Install required packages

First, the following Python packages need to be installed.

- [DeepSpeed](https://www.deepspeed.ai/)
- [Pyjnius](https://pyjnius.readthedocs.io) (LUKE)
- [PyICU](https://gitlab.pyicu.org/main/pyicu) (mLUKE)

**LUKE:**

```bash
poetry install --extras "pretraining opennlp"
```

**mLUKE:**

```bash
poetry install --extras "pretraining icu"
```

If you face trouble when installing these packages, please refer to the
documentation of the corresponding package.

## 2. Build a database from a Wikipedia dump

A Wikipedia dump file is converted to the database file using the
`build-dump-db` command. The dump file can be downloaded from
[Wikimedia Downloads](https://dumps.wikimedia.org/).

**LUKE:**

```bash
python luke/cli.py build-dump-db enwiki-latest-pages-articles.xml.bz2 enwiki.db
```

**mLUKE:**

## 3. Build an entity vocabulary

The entity vocabulary file can be built using the `build-entity-vocab` command.
The number of entities included in the vocabulary can be configured using the
`--vocab-size` option.

**LUKE:**

```bash
python luke/cli.py \
    build-entity-vocab \
    enwiki.db \
    luke_entity_vocab.jsonl \
    --vocab-size=500000
```

**mLUKE:**

## 4. Build a pretraining dataset

The processed dataset file can be built using the
`build-wikipedia-pretraining-dataset` command. `BASE_MODEL_NAME` should be
either `roberta-base` or `roberta-large` for LUKE and `xlm-roberta-base` or
`xlm-roberta-large` for mLUKE.

**LUKE:**

```bash
python luke/cli.py \
    build-wikipedia-pretraining-dataset \
    enwiki.db \
    <BASE_MODEL_NAME> \
    luke_entity_vocab.jsonl \
    luke_pretraining_dataset \
    --sentence-splitter=opennlp \
    --include-unk-entities
```

**mLUKE:**

## 5. Compute the number of training steps

Before starting to train the model, we need to compute the total number of
training steps using the `compute-total-training-steps` to configure learning
rate scheduler.

**LUKE:**

```bash
python luke/cli.py \
    compute-total-training-steps \
    --dataset-dir=luke_pretraining_dataset \
    --train-batch-size=2048 \
    --num-epochs=20
```

**mLUKE:**

## 6. Pretrain LUKE

The pretraining of LUKE is implemented based on
[DeepSpeed](https://www.deepspeed.ai/) and hyperparamters can be configured
using its [configuration JSON file](https://www.deepspeed.ai/docs/config-json/).

The configuration files corresponding to our publicized models (i.e.,
`luke-base`, `luke-large`, `mluke-base`, and `mluke-large`) are available in the
[pretraining_config](https://github.com/studio-ousia/luke/tree/master/pretraining_config)
directory. If you train the model using the configuration file, please set the
total number of training steps computed above in the `total_num_steps` property.

The pretraining of LUKE and mLUKE consists of two separate stages to stabilize
the training. Specifically, we update only the entity embeddings with large
learning late in the first stage, and train the entire model with small learning
rate in the second stage. When starting the second-stage training, the
checkpoint directory of the first stage model needs to be specified in the
`--resume-checkpoint-id` option.

**LUKE (stage 1):**

```bash
deepspeed \
    --num_gpus=<NUM_GPUS> \
    luke/pretraining/train.py \
    --output-dir=<OUTPUT_DIR> \
    --deepspeed-config-file=<DEEPSPEED_CONFIG_STAGE1_JSON_FILE> \
    --dataset-dir=luke_pretraining_dataset/ \
    --bert-model-name=<BASE_MODEL_NAME> \
    --num-epochs=<NUM_EPOCHS> \
    --fix-bert-weights
```

**LUKE (stage 2):**

```bash
deepspeed \
    --num_gpus=<NUM_GPUS> \
    luke/pretraining/train.py \
    --output-dir=<OUTPUT_DIR> \
    --deepspeed-config-file=<DEEPSPEED_CONFIG_STAGE2_JSON_FILE> \
    --dataset-dir=luke_pretraining_dataset/ \
    --bert-model-name=<BASE_MODEL_NAME> \
    --num-epochs=<NUM_EPOCHS> \
    --reset-optimization-states \
    --resume-checkpoint-id=<STAGE1_LAST_CHECKPOINT_DIR>
```

**mLUKE (stage 1):**

```bash
deepspeed \
    --num_gpus=<NUM_GPUS> \
    luke/pretraining/train.py \
    --output-dir=<OUTPUT_DIR> \
    --deepspeed-config-file=<DEEPSPEED_CONFIG_STAGE1_JSON_FILE> \
    --dataset-dir=mluke_pretraining_dataset/* \
    --bert-model-name=<BASE_MODEL_NAME> \
    --num-epochs=<NUM_EPOCHS> \
    --fix-bert-weights
```

**mLUKE (stage 2):**

```bash
deepspeed \
    --num_gpus=<NUM_GPUS> \
    luke/pretraining/train.py \
    --output-dir=<OUTPUT_DIR> \
    --deepspeed-config-file=<DEEPSPEED_CONFIG_STAGE2_JSON_FILE> \
    --dataset-dir=mluke_pretraining_dataset/* \
    --bert-model-name=<BASE_MODEL_NAME> \
    --num-epochs=<NUM_EPOCHS> \
    --reset-optimization-states \
    --resume-checkpoint-id=<STAGE1_LAST_CHECKPOINT_DIR>
```

## 7. Upload to HuggingFace Hub
