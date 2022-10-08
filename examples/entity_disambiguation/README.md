# Global Entity Disambiguation with BERT

This is the source code for our paper
[Global Entity Disambiguation with BERT](https://arxiv.org/abs/1909.00426).

This model addresses entity disambiguation based on LUKE using _local_
(word-based) and _global_ (entity-based) contextual information. The model is
trained by predicting randomly masked entities in Wikipedia, and achieves
state-of-the-art results on five standard entity disambiguation datasets:
AIDA-CoNLL, MSNBC, AQUAINT, ACE2004, and WNED-WIKI.

## Reproducing Experiments

- Model checkpoint file:
  [Link](https://drive.google.com/file/d/1aDx7PUsycyFGdHDF8RHCiq_LZ5hesWXX/view?usp=sharing)
- Dataset file:
  [Link](https://drive.google.com/file/d/1vjzrlp0uYtI6gjpnExdF3MtK21Orx9Lg/view?usp=sharing)

### Zero-shot evaluation

```bash
python examples/entity_disambiguation/evaluate.py
  --model-dir=<MODEL_DIR> \
  --dataset-dir=<DATASET_DIR> \
  --titles-file=<DATASET_DIR>/enwiki_20181220_titles.txt \
  --redirects-file=<DATASET_DIR>/enwiki_20181220_redirects.tsv \
  --inference-mode=global \
  --document-split-mode=per_mention
```

Please decompress the checkpoint file and dataset file and replace `<MODEL_DIR>`
and `<DATASET_DIR>` to the corresponding paths.

### Fine-tuning using CoNLL dataset

**Training:**

```bash
python examples/entity_disambiguation/train.py \
  --model-dir=<MODEL_DIR> \
  --dataset-dir=<DATASET_DIR> \
  --titles-file=<DATASET_DIR>/enwiki_20181220_titles.txt \
  --redirects-file=<DATASET_DIR>/enwiki_20181220_redirects.tsv \
  --output-dir=<OUTPUT_DIR>
```

**Evaluation:**

```bash
python examples/entity_disambiguation/evaluate.py
  --model-dir=<OUTPUT_DIR> \
  --dataset-dir=<DATASET_DIR> \
  --titles-file=<DATASET_DIR>/enwiki_20181220_titles.txt \
  --redirects-file=<DATASET_DIR>/enwiki_20181220_redirects.tsv \
  --inference-mode=global \
  --document-split-mode=per_mention
```

## Fast Inference

If you need fast inference speed, please set `--inference-mode=local` and
`--document-split-mode=simple`. This slightly degrades the performance, but the
code runs much faster.

```bash
python examples/entity_disambiguation/evaluate.py
  --model-dir=<MODEL_DIR> \
  --dataset-dir=<DATASET_DIR> \
  --titles-file=<DATASET_DIR>/enwiki_20181220_titles.txt \
  --redirects-file=<DATASET_DIR>/enwiki_20181220_redirects.tsv \
  --inference-mode=local \
  --document-split-mode=simple
```

## Training from Scratch

**1. Install required packages**

```bash
poetry install --extras "pretraining opennlp"
```

**2. Build database from Wikipedia dump**

```bash
python luke/cli.py build-dump-db \
  enwiki-latest-pages-articles.xml.bz2 \
  enwiki.db
```

The dump file can be downloaded from
[Wikimedia Downloads](https://dumps.wikimedia.org/).

**3. Create entity vocabulary**

```bash
export PYTHONPATH=examples/entity_disambiguation
python examples/entity_disambiguation/scripts/create_candidate_data.py \
  --db-file=enwiki.db \
  --dataset-dir=<DATASET_DIR> \
  --output-file=candidates.txt

python luke/cli.py build-entity-vocab \
  enwiki.db \
  entity_vocab.jsonl \
  --white-list=candidates.txt \
  --white-list-only
```

**4. Create training dataset**

```bash
python luke/cli.py build-wikipedia-pretraining-dataset \
  enwiki.db \
  bert-base-uncased \
  entity_vocab.jsonl \
  pretraining_dataset_for_ed \
  --sentence-splitter=opennlp \
  --include-unk-entities
```

**5. Train model**

Please see
[here](https://github.com/studio-ousia/luke/blob/master/pretraining.md) for the
details of the pretraining of LUKE.

The DeepSpeed configuration files corresponding to our publicized checkpoints
are available
[here](https://github.com/studio-ousia/luke/tree/entity_disambiguation/examples/entity_disambiguation/deepspeed_config).

_Stage 1:_

```bash
python luke/cli.py \
    compute-total-training-steps \
    --dataset-dir=pretraining_dataset_for_ed \
    --train-batch-size=2048 \
    --num-epochs=1
```

```bash
deepspeed \
    --num_gpus=<NUM_GPUS> \
    luke/pretraining/train.py \
    --output-dir=<OUTPUT_DIR> \
    --deepspeed-config-file=<DEEPSPEED_CONFIG_STAGE1_JSON_FILE> \
    --dataset-dir=pretraining_dataset_for_ed/ \
    --bert-model-name=bert-base-uncased \
    --num-epochs=1 \
    --masked-lm-prob=0.0 \
    --masked-entity-prob=0.3 \
    --fix-bert-weights
```

_Stage 2:_

```bash
python luke/cli.py \
    compute-total-training-steps \
    --dataset-dir=pretraining_dataset_for_ed \
    --train-batch-size=2048 \
    --num-epochs=6
```

```bash
deepspeed \
    --num_gpus=<NUM_GPUS> \
    luke/pretraining/train.py \
    --output-dir=<OUTPUT_DIR> \
    --deepspeed-config-file=<DEEPSPEED_CONFIG_STAGE2_JSON_FILE> \
    --dataset-dir=pretraining_dataset_for_ed/ \
    --bert-model-name=bert-base-uncased \
    --num-epochs=6 \
    --masked-lm-prob=0.0 \
    --masked-entity-prob=0.3 \
    --reset-optimization-states \
    --resume-checkpoint-id=<OUTPUT_DIR>/checkpoints/epoch1
```

### 6. Create Wikipedia data files

```bash
export PYTHONPATH=examples/entity_disambiguation
python examples/entity_disambiguation/scripts/create_title_data.py \
  --db-file=enwiki.db \
  --output-file=titles.txt

python examples/entity_disambiguation/scripts/create_redirect_data.py \
  --db-file=enwiki.db \
  --output-file=redirects.tsv
```

### 7. Convert checkpoint file

```bash
export PYTHONPATH=examples/entity_disambiguation
python examples/entity_disambiguation/scripts/convert_checkpoint.py \
    --checkpoint-file=<OUTPUT_DIR>/checkpoints/epoch6/mp_rank_00_model_states.pt \
    --metadata-file=<OUTPUT_DIR>/metadata.json \
    --entity-vocab-file=<OUTPUT_DIR>/entity_vocab.jsonl \
    --output-dir=<MODEL_DIR>
```

### 8. Evaluate model

```bash
python examples/entity_disambiguation/evaluate.py
  --model-dir=<MODEL_DIR> \
  --dataset-dir=<DATASET_DIR> \
  --titles-file=titles.txt \
  --redirects-file=redirects.txt \
  --inference-mode=global \
  --document-split-mode=per_mention
```

## Citation

If you find this work useful, please cite
[our paper](https://arxiv.org/abs/1909.00426):

```
@inproceedings{yamada-etal-2022-global-ed,
    title = "Global Entity Disambiguation with BERT",
    author = "Yamada, Ikuya  and
      Washio, Koki  and
      Shindo, Hiroyuki  and
      Matsumoto, Yuji",
    booktitle = "NAACL",
    year = "2022",
    publisher = "Association for Computational Linguistics"
}
```
