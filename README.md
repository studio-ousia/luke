<img src="resources/luke_logo.png" width="200" alt="LUKE">

[![CircleCI](https://circleci.com/gh/studio-ousia/luke.svg?style=svg&circle-token=49524bfde04659b8b54509f7e0f06ec3cf38f15e)](https://circleci.com/gh/studio-ousia/luke)

---

**LUKE** (**L**anguage **U**nderstanding with **K**nowledge-based
**E**mbeddings) is a new pretrained contextualized representation of words and
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

This repository contains the source code to pretrain the model and fine-tune it
to solve downstream tasks.

## News

**November 9, 2022: The large version of LUKE-Japanese is available**

The large version of LUKE-Japanese is available on the Hugging Face Model Hub:

- [luke-japanese-large](https://huggingface.co/studio-ousia/luke-japanese-large)
- [luke-japanese-large-lite](https://huggingface.co/studio-ousia/luke-japanese-large-lite)

This model achieves state-of-the-art results on three datasets in
[JGLUE](https://github.com/yahoojapan/JGLUE).

| Model                         | MARC-ja   | JSTS                | JNLI      | JCommonsenseQA |
| ----------------------------- | --------- | ------------------- | --------- | -------------- |
|                               | acc       | Pearson/Spearman    | acc       | acc            |
| **LUKE Japanese large**       | **0.965** | **0.932**/**0.902** | **0.927** | 0.893          |
| _Baselines:_                  |           |
| Tohoku BERT large             | 0.955     | 0.913/0.872         | 0.900     | 0.816          |
| Waseda RoBERTa large (seq128) | 0.954     | 0.930/0.896         | 0.924     | **0.907**      |
| Waseda RoBERTa large (seq512) | 0.961     | 0.926/0.892         | 0.926     | 0.891          |
| XLM RoBERTa large             | 0.964     | 0.918/0.884         | 0.919     | 0.840          |

**October 27, 2022: The Japanese version of LUKE is available**

The Japanese version of LUKE is now available on the Hugging Face Model Hub:

- [luke-japanese-base](https://huggingface.co/studio-ousia/luke-japanese-base)
- [luke-japanese-base-lite](https://huggingface.co/studio-ousia/luke-japanese-base-lite)

This model outperforms other base-sized models on four datasets in
[JGLUE](https://github.com/yahoojapan/JGLUE).

| Model                  | MARC-ja   | JSTS                | JNLI      | JCommonsenseQA |
| ---------------------- | --------- | ------------------- | --------- | -------------- |
|                        | acc       | Pearson/Spearman    | acc       | acc            |
| **LUKE Japanese base** | **0.965** | **0.916**/**0.877** | **0.912** | **0.842**      |
| _Baselines:_           |           |
| Tohoku BERT base       | 0.958     | 0.909/0.868         | 0.899     | 0.808          |
| NICT BERT base         | 0.958     | 0.910/0.871         | 0.902     | 0.823          |
| Waseda RoBERTa base    | 0.962     | 0.913/0.873         | 0.895     | 0.840          |
| XLM RoBERTa base       | 0.961     | 0.877/0.831         | 0.893     | 0.687          |

**April 13, 2022: The mLUKE fine-tuning code is available**

[The example code](examples) is updated. Now it is based on
[allennlp](https://github.com/allenai/allennlp) and
[transformers](https://github.com/huggingface/transformers). You can reproduce
the experiments in the [LUKE](https://arxiv.org/abs/2010.01057) and
[mLUKE](https://arxiv.org/abs/2110.08151) papers with this implementation. For
the details, please see `README.md` under each example directory. The older code
used in [the LUKE paper](https://arxiv.org/abs/2010.01057) has been moved to
[`examples/legacy`](examples/legacy).

**April 13, 2022: The detailed instructions for pretraining LUKE models are
available**

For those interested in pretraining LUKE models, we explain how to prepare
datasets and run the pretraining code on [`pretraining.md`](pretraining.md).

**November 24, 2021: Entity disambiguation example is available**

The example code of entity disambiguation based on LUKE has been added to this
repository. This model was originally proposed in
[our paper](https://arxiv.org/abs/1909.00426), and achieved state-of-the-art
results on five standard entity disambiguation datasets: AIDA-CoNLL, MSNBC,
AQUAINT, ACE2004, and WNED-WIKI.

For further details, please refer to
[`examples/entity_disambiguation`](examples/entity_disambiguation).

**August 3, 2021: New example code based on Hugging Face Transformers and
AllenNLP is available**

New fine-tuning examples of three downstream tasks, i.e., _NER_, _relation
classification_, and _entity typing_, have been added to LUKE. These examples
are developed based on Hugging Face Transformers and AllenNLP. The fine-tuning
models are defined using simple AllenNLP's Jsonnet config files!

The example code is available in [`examples`](examples).

**May 5, 2021: LUKE is added to Hugging Face Transformers**

LUKE has been added to the
[master branch of the Hugging Face Transformers library](https://github.com/huggingface/transformers).
You can now solve entity-related tasks (e.g., named entity recognition, relation
classification, entity typing) easily using this library.

For example, the LUKE-large model fine-tuned on the TACRED dataset can be used
as follows:

```python
from transformers import LukeTokenizer, LukeForEntityPairClassification
model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
text = "Beyoncé lives in Los Angeles."
entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = int(logits[0].argmax())
print("Predicted class:", model.config.id2label[predicted_class_idx])
# Predicted class: per:cities_of_residence
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
poetry install

# If you want to run pretraining for LUKE
poetry install --extras "pretraining opennlp"
# If you want to run pretraining for mLUKE
poetry install --extras "pretraining icu"
```

The virtual environment automatically created by Poetry can be activated by
`poetry shell`.

**A note on installing `torch`**

The pytorch installed via `poetry install` does not necessarily match your
hardware. In such case, see [the official site](https://pytorch.org/) and
reinstall the correct version with the `pip` command.

```bash
poetry run pip3 uninstall torch torchvision torchaudio
# Example for Linux with CUDA 11.3
poetry run pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

## Released Models

Our pretrained models can be used with the
[transformers](https://github.com/huggingface/transformers) library. The model
documentations can be found in the following links:
[LUKE](https://huggingface.co/docs/transformers/main/en/model_doc/luke) and
[mLUKE](https://huggingface.co/docs/transformers/main/en/model_doc/mluke).

Currently, the following models are available on
[the Hugging Face Model Hub](https://huggingface.co/models).

|           Name            |                                         model_name                                          | Entity Vocab Size | Params |
| :-----------------------: | :-----------------------------------------------------------------------------------------: | :---------------: | :----: |
|      **LUKE (base)**      |           [studio-ousia/luke-base](https://huggingface.co/studio-ousia/luke-base)           |       500K        | 253 M  |
|     **LUKE (large)**      |          [studio-ousia/luke-large](https://huggingface.co/studio-ousia/luke-large)          |       500K        | 484 M  |
|     **mLUKE (base)**      |          [studio-ousia/mluke-base](https://huggingface.co/studio-ousia/mluke-base)          |       1.2M        | 586 M  |
|     **mLUKE (large)**     |         [studio-ousia/mluke-large](https://huggingface.co/studio-ousia/mluke-large)         |       1.2M        | 868 M  |
| **LUKE Japanese (base)**  |  [studio-ousia/luke-japanese-base](https://huggingface.co/studio-ousia/luke-japanese-base)  |       570K        | 281 M  |
| **LUKE Japanese (large)** | [studio-ousia/luke-japanese-large](https://huggingface.co/studio-ousia/luke-japanese-large) |       570K        | 562 M  |

### Lite Models

The entity embeddings cause a large memory footprint as they contain all the
Wikipedia entities that we used in pretraining. However, in some downstream
tasks (e.g., entity typing, named entity recognition, and relation
classification), we only need special entity embeddings such as `[MASK]`. Also,
you may want to only use the word representations.

With such use-cases in mind, to make our models easier to use, we have uploaded
lite models only with special entity embeddings. These models perform exactly
the same as the full models but have much fewer parameters, which enable
fine-tuning the model with small GPUs.

|           Name            |                                              model_name                                               | Params |
| :-----------------------: | :---------------------------------------------------------------------------------------------------: | :----: |
|      **LUKE (base)**      |           [studio-ousia/luke-base-lite](https://huggingface.co/studio-ousia/luke-base-lite)           | 125 M  |
|     **LUKE (large)**      |          [studio-ousia/luke-large-lite](https://huggingface.co/studio-ousia/luke-large-lite)          | 356 M  |
|     **mLUKE (base)**      |          [studio-ousia/mluke-base-lite](https://huggingface.co/studio-ousia/mluke-base-lite)          | 279 M  |
|     **mLUKE (large)**     |         [studio-ousia/mluke-large-lite](https://huggingface.co/studio-ousia/mluke-large-lite)         | 561 M  |
| **LUKE Japanese (base)**  |  [studio-ousia/luke-japanese-base-lite](https://huggingface.co/studio-ousia/luke-japanese-base-lite)  | 134 M  |
| **LUKE Japanese (large)** | [studio-ousia/luke-japanese-large-lite](https://huggingface.co/studio-ousia/luke-japanese-large-lite) | 415 M  |

## Fine-tuning LUKE models

We release the fine-tuning code based on
[allennlp](https://github.com/allenai/allennlp) and
[transformers](https://github.com/huggingface/transformers) under
[`examples`](examples). You can run fine-tuning experiments very easily with
pre-defined config files and the `allennlp train` command. For the details and
example commands for each task, please see the task directory under
[`examples`](examples).

## Pretraining LUKE models

The detailed instructions for pretraining luke models can be found on
[`pretraining.md`](pretraining.md).

## Citation

If you use LUKE in your work, please cite the
[original paper](https://aclanthology.org/2020.emnlp-main.523/).

```
@inproceedings{yamada-etal-2020-luke,
    title = "{LUKE}: Deep Contextualized Entity Representations with Entity-aware Self-attention",
    author = "Yamada, Ikuya  and
      Asai, Akari  and
      Shindo, Hiroyuki  and
      Takeda, Hideaki  and
      Matsumoto, Yuji",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.523",
    doi = "10.18653/v1/2020.emnlp-main.523",
}
```

For mLUKE, please cite
[this paper](https://aclanthology.org/2022.acl-long.505/).

```
@inproceedings{ri-etal-2022-mluke,
    title = "m{LUKE}: {T}he Power of Entity Representations in Multilingual Pretrained Language Models",
    author = "Ri, Ryokan  and
      Yamada, Ikuya  and
      Tsuruoka, Yoshimasa",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.505",
}
```
