# Pretraining LUKE

This document describes how to pretrain LUKE using a Wikipedia dump file.

## 1. Install required packages

**LUKE:**

```bash
poetry install --extras "pretraining opennlp"
```

**mLUKE:**

```bash
poetry install --extras "pretraining icu"
```

## 2. Build a database from a Wikipedia dump

**LUKE:**

```bash
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
python luke/cli.py build-dump-db enwiki-latest-pages-articles.xml.bz2 enwiki.lmdb
```

**mLUKE:**

## 3. Build an entity vocabulary

**LUKE:**

```bash
python luke/cli.py \
    build-entity-vocab \
    enwiki.lmdb \
    enwiki_entity_vocab.jsonl \
    --vocab-size=500000
```

**mLUKE:**

The size of the entity vocabulary can be configured using `--vocab-size` option.

## 4. Build a pretraining dataset

**LUKE:**

```bash
python luke/cli.py \
    build-wikipedia-pretraining-dataset \
    enwiki.lmdb \
    <BASE_MODEL_NAME> \
    enwiki_entity_vocab.jsonl \
    enwiki_pretraining_dataset \
    --sentence-splitter=opennlp \
    --include-unk-entities
```

`BASE_MODEL_NAME` should be either `roberta-base` or `roberta-large`.

**mLUKE:**

## 5. Compute the number of training steps

**LUKE:**

```bash
python luke/cli.py \
    compute-total-training-steps \
    --dataset-dir=enwiki_pretraining_dataset \
    --train-batch-size=2048 \
    --num-epochs=20
```

**mLUKE:**

## 5. Pretrain LUKE

The pretraining of LUKE is implemented based on
[Deepspeed](https://www.deepspeed.ai/) and the hyperparamters can be configured
using its [Configuration JSON](https://www.deepspeed.ai/docs/config-json/).

The configuration files of our pretrained models are available in the
[pretraining_config](https://github.com/studio-ousia/luke/tree/master/pretraining_config)
directory.

The pretraining of LUKE and mLUKE consists of two separate stages. We update
only the entity embeddings in the first stage, and train the entire model in the
second stage.

**LUKE (stage 1):**

```bash
deepspeed \
    --num_gpus=<NUM_GPUS> \
    luke/pretraining/train.py \
    --output-dir=<OUTPUT_DIR> \
    --deepspeed-config-file=<DEEPSPEED_CONFIG_JSON_FILE> \
    --dataset-dir=enwiki_pretraining_dataset/ \
    --bert-model-name=<BASE_MODEL_NAME> \
    --num-epochs=<NUM_EPOCHS> \
    --fix-bert-weights \
```

**LUKE (stage 2):**

```bash
deepspeed \
    --num_gpus=<NUM_GPUS> \
    luke/pretraining/train.py \
    --output-dir=<OUTPUT_DIR> \
    --deepspeed-config-file=<DEEPSPEED_CONFIG_JSON_FILE> \
    --dataset-dir=enwiki_pretraining_dataset/ \
    --bert-model-name=<BASE_MODEL_NAME> \
    --num-epochs=<NUM_EPOCHS> \
    --reset-optimization-states \
    --resume-checkpoint-id=<STAGE1_LAST_CHECKPOINT_DIR> \
```
