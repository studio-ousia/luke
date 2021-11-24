<img src="resources/luke_logo.png" width="200" alt="LUKE">

[![CircleCI](https://circleci.com/gh/studio-ousia/luke.svg?style=svg&circle-token=49524bfde04659b8b54509f7e0f06ec3cf38f15e)](https://circleci.com/gh/studio-ousia/luke)

---

**LUKE** (**L**anguage **U**nderstanding with **K**nowledge-based
**E**mbeddings) is a new pre-trained contextualized representation of words and
entities based on transformer. It was proposed in our paper
[LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://arxiv.org/abs/2010.01057).
It achieves state-of-the-art results on important NLP benchmarks including
**[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)** (extractive
question answering),
**[CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)** (named entity
recognition), **[ReCoRD](https://sheng-z.github.io/ReCoRD-explorer/)**
(cloze-style question answering),
**[TACRED](https://nlp.stanford.edu/projects/tacred/)** (relation
classification), and
**[Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html)**
(entity typing).

This repository contains the source code to pre-train the model and fine-tune it
to solve downstream tasks.

## News

**November 24, 2021: Entity disambiguation example is available**

The example code of entity disambiguation based on LUKE has been added to this
repository. This model was originally proposed in
[our paper](https://arxiv.org/abs/1909.00426), and achieved state-of-the-art
results on five standard entity disambiguation datasets: AIDA-CoNLL, MSNBC,
AQUAINT, ACE2004, and WNED-WIKI.

For further details, please refer to the
[example directory](https://github.com/studio-ousia/luke/tree/master/examples/entity_disambiguation).

**August 3, 2021: New example code based on Hugging Face Transformers and
AllenNLP is available**

New fine-tuning examples of three downstream tasks, i.e., _NER_, _relation
classification_, and _entity typing_, have been added to LUKE. These examples
are developed based on Hugging Face Transformers and AllenNLP. The fine-tuning
models are defined using simple AllenNLP's Jsonnet config files!

The example code is available in the
[examples_allennlp directory](https://github.com/studio-ousia/luke/tree/master/examples_allennlp).

**May 5, 2021: LUKE is added to Hugging Face Transformers**

LUKE has been added to the
[master branch of the Hugging Face Transformers library](https://github.com/huggingface/transformers).
You can now solve entity-related tasks (e.g., named entity recognition, relation
classification, entity typing) easily using this library.

For example, the LUKE-large model fine-tuned on the TACRED dataset can be used
as follows:

```python
>>> from transformers import LukeTokenizer, LukeForEntityPairClassification
>>> model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
>>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
>>> text = "Beyoncé lives in Los Angeles."
>>> entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
>>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> predicted_class_idx = int(logits[0].argmax())
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: per:cities_of_residence
```

We also provide the following three Colab notebooks that show how to reproduce
our experimental results on CoNLL-2003, TACRED, and Open Entity datasets using
the library:

- [Reproducing experimental results of LUKE on CoNLL-2003 Using Hugging Face Transformers](https://colab.research.google.com/github/studio-ousia/luke/blob/master/notebooks/huggingface_conll_2003.ipynb)
- [Reproducing experimental results of LUKE on TACRED Using Hugging Face Transformers](https://colab.research.google.com/github/studio-ousia/luke/blob/master/notebooks/huggingface_tacred.ipynb)
- [Reproducing experimental results of LUKE on Open Entity Using Hugging Face Transformers](https://colab.research.google.com/github/studio-ousia/luke/blob/master/notebooks/huggingface_open_entity.ipynb)

Please refer to the
[official documentation](https://huggingface.co/transformers/master/model_doc/luke.html)
for further details.

**November 5, 2021: LUKE-500K (base) model**

We released LUKE-500K (base), a new pretrained LUKE model which is smaller than
existing LUKE-500K (large). The experimental results of the LUKE-500K (base) and
LUKE-500K (large) on SQuAD v1 and CoNLL-2003 are shown as follows:

| Task                          | Dataset                                                      | Metric | LUKE-500K (base) | LUKE-500K (large) |
| ----------------------------- | ------------------------------------------------------------ | ------ | ---------------- | ----------------- |
| Extractive Question Answering | [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)    | EM/F1  | 86.1/92.3        | 90.2/95.4         |
| Named Entity Recognition      | [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) | F1     | 93.3             | 94.3              |

We tuned only the batch size and learning rate in the experiments based on
LUKE-500K (base).

## Comparison with State-of-the-Art

LUKE outperforms the previous state-of-the-art methods on five important NLP
tasks:

| Task                           | Dataset                                                                      | Metric | LUKE-500K (large) | Previous SOTA                                                             |
| ------------------------------ | ---------------------------------------------------------------------------- | ------ | ----------------- | ------------------------------------------------------------------------- |
| Extractive Question Answering  | [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)                    | EM/F1  | **90.2**/**95.4** | 89.9/95.1 ([Yang et al., 2019](https://arxiv.org/abs/1906.08237))         |
| Named Entity Recognition       | [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)                 | F1     | **94.3**          | 93.5 ([Baevski et al., 2019](https://arxiv.org/abs/1903.07785))           |
| Cloze-style Question Answering | [ReCoRD](https://sheng-z.github.io/ReCoRD-explorer/)                         | EM/F1  | **90.6**/**91.2** | 83.1/83.7 ([Li et al., 2019](https://www.aclweb.org/anthology/D19-6011/)) |
| Relation Classification        | [TACRED](https://nlp.stanford.edu/projects/tacred/)                          | F1     | **72.7**          | 72.0 ([Wang et al. , 2020](https://arxiv.org/abs/2002.01808))             |
| Fine-grained Entity Typing     | [Open Entity](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html) | F1     | **78.2**          | 77.6 ([Wang et al. , 2020](https://arxiv.org/abs/2002.01808))             |

These numbers are reported in
[our EMNLP 2020 paper](https://arxiv.org/abs/2010.01057).

## Installation

LUKE can be installed using [Poetry](https://python-poetry.org/):

```bash
$ poetry install
```

The virtual environment automatically created by Poetry can be activated by
`poetry shell`.

## Released Models

We initially release the pre-trained model with 500K entity vocabulary based on
the `roberta.large` model.

| Name                  | Base Model                                                                                          | Entity Vocab Size | Params | Download                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------------- | ----------------- | ------ | ------------------------------------------------------------------------------------------ |
| **LUKE-500K (base)**  | [roberta.base](https://github.com/pytorch/fairseq/tree/master/examples/roberta#pre-trained-models)  | 500K              | 253 M  | [Link](https://drive.google.com/file/d/17JvBfXTMuXHX_00yq6kXUDB6OJStfSK_/view?usp=sharing) |
| **LUKE-500K (large)** | [roberta.large](https://github.com/pytorch/fairseq/tree/master/examples/roberta#pre-trained-models) | 500K              | 483 M  | [Link](https://drive.google.com/file/d/1S7smSBELcZWV7-slfrb94BKcSCCoxGfL/view?usp=sharing) |

## Reproducing Experimental Results

The experiments were conducted using Python3.6 and PyTorch 1.2.0 installed on a
server with a single or eight NVidia V100 GPUs. We used
[NVidia's PyTorch Docker container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
19.02. For computational efficiency, we used mixed precision training based on
APEX library which can be installed as follows:

```bash
$ git clone https://github.com/NVIDIA/apex.git
$ cd apex
$ git checkout c3fad1ad120b23055f6630da0b029c8b626db78f
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```

The APEX library is not needed if you do not use `--fp16` option or reproduce
the results based on the trained checkpoint files.

The commands that reproduce the experimental results are provided as follows:

### Entity Typing on Open Entity Dataset

**Dataset:** [Link](https://github.com/thunlp/ERNIE)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/10F6tzx0oPG4g-PeB0O1dqpuYtfiHblZU/view?usp=sharing)

**Using the checkpoint file:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    entity-typing run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    entity-typing run \
    --data-dir=<DATA_DIR> \
    --train-batch-size=2 \
    --gradient-accumulation-steps=2 \
    --learning-rate=1e-5 \
    --num-train-epochs=3 \
    --fp16
```

### Relation Classification on TACRED Dataset

**Dataset:** [Link](https://nlp.stanford.edu/projects/tacred/)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/10XSaQRtQHn13VB_6KALObvok6hdXw7yp/view?usp=sharing)

**Using the checkpoint file:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    relation-classification run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    relation-classification run \
    --data-dir=<DATA_DIR> \
    --train-batch-size=4 \
    --gradient-accumulation-steps=8 \
    --learning-rate=1e-5 \
    --num-train-epochs=5 \
    --fp16
```

### Named Entity Recognition on CoNLL-2003 Dataset

**Dataset:** [Link](https://www.clips.uantwerpen.be/conll2003/ner/)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/10VFEHXMiJGQvD62QbHa8C8XYSeAIt_CP/view?usp=sharing)

**Using the checkpoint file:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    ner run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
$ python -m examples.cli\
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    ner run \
    --data-dir=<DATA_DIR> \
    --train-batch-size=2 \
    --gradient-accumulation-steps=4 \
    --learning-rate=1e-5 \
    --num-train-epochs=5 \
    --fp16
```

### Cloze-style Question Answering on ReCoRD Dataset

**Dataset:** [Link](https://sheng-z.github.io/ReCoRD-explorer/)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/10LuPIQi-HslZs_BgHxSnitGe2tw_anZp/view?usp=sharing)

**Using the checkpoint file:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    entity-span-qa run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
$ python -m examples.cli \
    --num-gpus=8 \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    entity-span-qa run \
    --data-dir=<DATA_DIR> \
    --train-batch-size=1 \
    --gradient-accumulation-steps=4 \
    --learning-rate=1e-5 \
    --num-train-epochs=2 \
    --fp16
```

### Extractive Question Answering on SQuAD 1.1 Dataset

**Dataset:** [Link](https://rajpurkar.github.io/SQuAD-explorer/)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/1097QicHAVnroVVw54niPXoY-iylGNi0K/view?usp=sharing)\
**Wikipedia data files (compressed):**
[Link](https://drive.google.com/file/d/129tDJ3ev6IdbJiKOmO6GTgNANunhO_vt/view?usp=sharing)

**Using the checkpoint file:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    reading-comprehension run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-negative \
    --wiki-link-db-file=enwiki_20160305.pkl \
    --model-redirects-file=enwiki_20181220_redirects.pkl \
    --link-redirects-file=enwiki_20160305_redirects.pkl \
    --no-train
```

**Fine-tuning the model:**

```bash
$ python -m examples.cli \
    --num-gpus=8 \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    reading-comprehension run \
    --data-dir=<DATA_DIR> \
    --no-negative \
    --wiki-link-db-file=enwiki_20160305.pkl \
    --model-redirects-file=enwiki_20181220_redirects.pkl \
    --link-redirects-file=enwiki_20160305_redirects.pkl \
    --train-batch-size=2 \
    --gradient-accumulation-steps=3 \
    --learning-rate=15e-6 \
    --num-train-epochs=2 \
    --fp16
```

## Citation

If you use LUKE in your work, please cite the
[original paper](https://arxiv.org/abs/2010.01057):

```
@inproceedings{yamada2020luke,
  title={LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention},
  author={Ikuya Yamada and Akari Asai and Hiroyuki Shindo and Hideaki Takeda and Yuji Matsumoto},
  booktitle={EMNLP},
  year={2020}
}
```

## Contact Info

Please submit a GitHub issue or send an e-mail to Ikuya Yamada
(`ikuya@ousia.jp`) for help or issues using LUKE.
