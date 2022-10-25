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

Create the dataset for the 24 languages.
```bash
for l in ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh
python luke/cli.py build-dump-db "${l}wiki-latest-pages-articles.xml.bz2" "${l}wiki.db"
```

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


To build a multilingual vocabulary, you need an interwiki DB to map same entities across languages into the same ids.
Download the latest wikidata dump from [here](https://dumps.wikimedia.org/wikidatawiki/entities), and then build the interwiki DB by `python luke/cli.py build-interwiki-db`.

Example
```bash
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2

python luke/cli.py build-interwiki-db latest-all.json.bz2 interwiki.db
```

Create entity vocabularies for each language and then combine them with the interwiki DB.
```bash
for l in ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh
python luke/cli.py  build-entity-vocab "${l}wiki.db" "mluke_entity_vocab_${l}.jsonl" "--language ${l}" 


COMMAND="python luke/cli.py build_multilingual_entity_vocab -i interwiki.db -o mluke_entity_vocab.jsonl --vocab-size 1200000 --min-num-languages 3"
# add options by for loop because there are so many..
for l in ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh
COMMAND=$COMMAND+" -v mluke_entity_vocab_${l}.jsonl"
eval $COMMAND
```

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
```bash
for l in ar bn de nl el en es fi fr hi id it ja ko pl pt ru sv sw te th tr vi zh
python luke/cli.py \
    build-wikipedia-pretraining-dataset \
    "${l}wiki.db" \
    <BASE_MODEL_NAME> \
    mluke_entity_vocab.jsonl \
    "mluke_pretraining_dataset/${l}" \
    --sentence-splitter=$l \
    --language $l 
```

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
```bash
python luke/cli.py \
    compute-total-training-steps \
    --dataset-dir="mluke_pretraining_dataset/*" \
    --train-batch-size=2048 \
    --num-epochs=20
```

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
    --subword-masking　 \
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
    --subword-masking　 \
    --resume-checkpoint-id=<STAGE1_LAST_CHECKPOINT_DIR>
```

## 7. Use the pretrained model with HuggingFace Transformers
The pretrained model can be used with the [transformers](https://github.com/huggingface/transformers) library after converting the checkpoint weights and metadata to the appropriate format.
Specify the saved files and choose the appropriate tokenizer class (`--tokenizer-class`) from `LukeTokenizer` or `MLukeTokenizer`.

The option `--set-entity-aware-attention-default` specifies whether the weights for the entity-aware attention is used by default when later loading the model from Transformers. 
Note that even when you didn't use the entity-aware attention during pretraining, the weights are copied from the standard attention, and you can still use the entity-aware attention during fine-tuning.

You can also specify `--remove-entity-embeddings` option to make a lite-weight model without entity embeddings but still with knowledge-enhanced word representations.
```bash
python luke/cli.py \
    convert-luke-to-huggingface-model \ 
    --checkpoint-path=<OUTPUT_DIR>/checkpoints/epoch20/mp_rank_00_model_states.pt \
    --metadata-path=<OUTPUT_DIR>/metadata.json  \
    --entity-vocab-path=<OUTPUT_DIR>/entity_vocab.jsonl \ 
    --transformers-model-save-path=<TRANSFORMER_MODEL_SAVE_PATH> \ 
    --tokenizer-class=<TOKENIZER_CLASS> \
    --set-entity-aware-attention-default=false
```

Then you can load the model with the Transformers library.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained(TRANSFORMER_MODEL_SAVE_PATH)
```

Also, you can upload the model to the Hugging Face Hub by following the instructions [here](https://huggingface.co/docs/hub/adding-a-model).
