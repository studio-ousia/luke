# Global Entity Disambiguation with Pretrained Contextualized Embeddings of Words and Entities

This directory contains the source code for our paper
[Global Entity Disambiguation with Pretrained Contextualized Embeddings of Words and Entities](https://arxiv.org/abs/1909.00426).

The proposed model addresses entity disambiguation based on LUKE using _local_
(word-based) and _global_ (entity-based) contextual information. The model is
fine-tuned by predicting randomly masked entities in Wikipedia. This model
achieves state-of-the-art results on five standard entity disambiguation
datasets: AIDA-CoNLL, MSNBC, AQUAINT, ACE2004, and WNED-WIKI.

## Reproducing Experiments

- Model checkpoint file:
  [Link](https://drive.google.com/file/d/1BTf9XM83tWrq9VOXqj9fXlGm2mP5DNRF/view?usp=sharing)
- Dataset:
  [Link](https://drive.google.com/file/d/1vjzrlp0uYtI6gjpnExdF3MtK21Orx9Lg/view?usp=sharing)

**Zero-shot evaluation of the trained model:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_ed.tar.gz
    --output-dir=<OUTPUT_DIR> \
    entity-disambiguation run
    --data-dir=<DATA_DIR>
    --no-train \
    --do-eval
```

**Fine-tuning the model using the CoNLL dataset:**

```bash
$ python -m examples.cli \
    --model-file=luke_large_ed.tar.gz \
    entity-disambiguation run \
    --data-dir=data/entity_disambiguation \
    --learning-rate=2e-5 \
    --adam-b2=0.999 \
    --max-grad-norm=1.0 \
    --warmup-proportion=0.1 \
    --train-batch-size=2 \
    --gradient-accumulation-steps=8 \
    --do-train \
    --do-eval
```

## Citation

If you find this work useful, please cite
[our paper](https://arxiv.org/abs/1909.00426):

```
@article{yamada2019global,
  title={Global Entity Disambiguation with Pretrained Contextualized Embeddings of Words and Entities},
  author={Ikuya Yamada and Koki Washio and Hiroyuki Shindo and Yuji Matsumoto},
  journal={arXiv preprint arXiv:1909.00426},
  year={2019}
}
```
