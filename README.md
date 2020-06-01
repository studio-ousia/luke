# LUKE

LUKE is a new pretrained contextualized representation of words and entities based on [bidirectional transformer](https://arxiv.org/abs/1706.03762).
The model is designed to effectively address various entity-related tasks including entity typing, relation classification, named entity recognition, and question answering.
This repository contains the source code to pretrain the model, and fine-tune the model to solve downstream tasks.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
transformers-cli download roberta-large
```

The Apex library needs to be installed if you use the mixed precision training activated by the `--fp16` option.

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..
```

## Downloading Pretrained Model

The pretrained model and several data files required to reproduce the experimental results can be downloaded from [this link](https://drive.google.com/file/d/1c7uodwgcHQ68svkzlsWkhw-AbOv4Tt6e/view?usp=sharing).

```bash
mkdir model
tar xvzf luke_20200528.tar.gz -C model
```

## Reproducing Experimental Results

The following commands should reproduce the results reported in the paper.
We conducted experiments using Python3.6 installed on a server with a single or eight NVidia V100 GPUs.
The dataset needs to properly placed in the DATA\_DIR.

### Entity Typing on Open Entity Dataset

```bash
python -m examples.cli --data-dir=DATA_DIR --weights-file=model/luke.bin entity-typing run --fp16 --train-batch-size=2 --gradient-accumulation-steps=2 --learning-rate=1e-5 --num-train-epochs=3 --word-entity-query
```

### Relation Classification on TACRED Dataset

```bash
python -m examples.cli --data-dir=DATA_DIR --weights-file=model/luke.bin relation-classification run --fp16 --train-batch-size=4 --gradient-accumulation-steps=8 --learning-rate=1e-5 --num-train-epochs=5 --word-entity-query
```

### Named Entity Recognition on CoNLL-2003 Dataset

```bash
python -m examples.cli --data-dir=DATA_DIR --weights-file=model/luke.bin ner run --fp16 --train-batch-size=2 --gradient-accumulation-steps=4 --learning-rate=1e-5 --num-train-epochs=5 --word-entity-query
```

### Cloze-style Question Answering on ReCoRD Dataset

```bash
python -m examples.cli --data-dir=DATA_DIR --weights-file=model/luke.bin --num-gpus=8 entity-span-qa run --fp16 --train-batch-size=1 --gradient-accumulation-steps=4 --learning-rate=1e-5 --num-train-epochs=2 --word-entity-query
```

### Extractive Question Answering on SQuAD 1.1 Dataset

```bash
python -m examples.cli --num-gpus=8 --data-dir=DATA_DIR --weights-file=model/luke.bin --mention-db-file=model/enwiki_20160305_mention_lp0.0.pkl --model-redirects-file=model/enwiki_20181220_redirects.pkl --link-redirects-file=model/enwiki_20160305_redirects.pkl reading-comprehension run --no-negative --fp16 --train-batch-size=2 --gradient-accumulation-steps=3 --learning-rate=15e-6 --num-train-epochs=2 --word-entity-query
```
