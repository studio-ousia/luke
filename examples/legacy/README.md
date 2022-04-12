## Reproducing Experimental Results

The experiments were conducted using Python3.6 and PyTorch 1.2.0 installed on a
server with a single or eight NVidia V100 GPUs. We used
[NVidia's PyTorch Docker container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
19.02. For computational efficiency, we used mixed precision training based on
APEX library which can be installed as follows:

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout c3fad1ad120b23055f6630da0b029c8b626db78f
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```

The APEX library is not needed if you do not use `--fp16` option or reproduce
the results based on the trained checkpoint files.

The commands that reproduce the experimental results are provided as follows:

### Entity Typing on Open Entity Dataset

**Dataset:** [Link](https://github.com/thunlp/ERNIE)\
**Checkpoint file (compressed):** [Link](https://drive.google.com/file/d/10F6tzx0oPG4g-PeB0O1dqpuYtfiHblZU/view?usp=sharing)

**Prepare the dataset**
```Bash
gdown --id 1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im
tar xzf data.tar.gz
```

**Using the checkpoint file:**

```bash
python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    entity-typing run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
python -m examples.cli \
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
python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    relation-classification run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
python -m examples.cli \
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
python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    ner run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
python -m examples.cli\
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
python -m examples.cli \
    --model-file=luke_large_500k.tar.gz \
    --output-dir=<OUTPUT_DIR> \
    entity-span-qa run \
    --data-dir=<DATA_DIR> \
    --checkpoint-file=<CHECKPOINT_FILE> \
    --no-train
```

**Fine-tuning the model:**

```bash
python -m examples.cli \
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
python -m examples.cli \
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
python -m examples.cli \
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