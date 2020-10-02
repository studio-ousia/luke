# LUKE

[![CircleCI](https://circleci.com/gh/studio-ousia/luke.svg?style=svg&circle-token=49524bfde04659b8b54509f7e0f06ec3cf38f15e)](https://circleci.com/gh/studio-ousia/luke)

LUKE is a new _knowledge-enhanced_ contextualized representation of words and
entities based on [transformer](https://arxiv.org/abs/1706.03762). LUKE is
designed to effectively address entity-related tasks including entity typing,
relation classification, named entity recognition, and question answering. This
model achieved state-of-the-art results on popular datasets including SQuAD 1.1
(extractive question answering), CoNLL-2003 (named entity recognition), ReCoRD
(cloze-style question answering), TACRED (relation classification), and Open
Entity (entity typing). This repository contains the source code to pretrain the
model, and fine-tune the model to solve downstream tasks.

## Installation

```bash
poetry install
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout c3fad1ad120b23055f6630da0b029c8b626db78f
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..
```

## Downloading Pretrained Model

The pretrained model and data files required to reproduce the experimental
results can be downloaded from
[this link](https://drive.google.com/file/d/1c7uodwgcHQ68svkzlsWkhw-AbOv4Tt6e/view?usp=sharing).

```bash
mkdir model
tar xvzf luke_20200528.tar.gz -C model
```

## Reproducing Experimental Results

(coming soon)
