# Relation Classification
In this code, you can experiment with the task of relation classification with several datasets. Currently, we support the following datasets.

## Datasets 

#####  English 
* [TACRED](https://www.aclweb.org/anthology/D17-1004/)
* [KBP37](https://arxiv.org/abs/1508.01006)

#####  Cross-lingual Evaluation
* [RELX](https://www.aclweb.org/anthology/2020.findings-emnlp.32/)

##### Download datasets
```bash
cd data
git clone https://github.com/zhangdongxu/kbp37.git
git clone https://github.com/boun-tabi/RELX.git
```

For the TACRED dataset, you need to access through [LDC](https://catalog.ldc.upenn.edu/LDC2018T24).

## Reproduce the checkpoint result
### LUKE-large
```bash
# Reproduce the result of studio-ousia/luke-large-finetuned-tacred.
poetry run python examples/relation_classification/evaluate_transformers_checkpoint.py tacred data/tacred/test.json studio-ousia/luke-large-finetuned-tacred --cuda-device 0
# expected results:
# {'accuracy': 0.8887742601070346, 'macro_fscore': 0.5886632942601121, 'micro_fscore': 0.7267450297489478}.
```

### mLUKE-large
```bash
# Reproduce the result of studio-ousia/mluke-large-lite-finetuned-kbp37
poetry run python examples/relation_classification/evaluate_transformers_checkpoint.py kbp37 data/RELX/Datasets/RELX/RELX_es.txt studio-ousia/mluke-large-lite-finetuned-kbp37 --cuda-device 0
# expected results:
# {'accuracy': 0.651394422310757, 'macro_fscore': 0.6782825333267248, 'micro_fscore': 0.6848072562358276}
```

## Training
We configure some parameters through environmental variables.  
Note that you need provide what dataset you are dealing with through `TASK`.

### Monolingual (Tacred)
```bash
export DATASET="tacred";
export TRAIN_DATA_PATH="data/tacred/train.json";
export VALIDATION_DATA_PATH="data/tacred/dev.json";

# train LUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-base";
poetry run allennlp train examples/relation_classification/configs/transformers_luke_with_entity_aware_attention.jsonnet -s results/relation_classification/luke-base --include-package examples -o '{"trainer": {"cuda_device": 0}}'

# you can also fine-tune models from the BERT family
export TRANSFORMERS_MODEL_NAME="roberta-base";
poetry run allennlp train examples/relation_classification/configs/transformers.jsonnet  -s results/relation_classification/roberta-base --include-package examples
```

### Multilingual (KBP37)
```bash
export DATASET="kbp37";
export TRAIN_DATA_PATH="data/kbp37/train.txt";
export VALIDATION_DATA_PATH="data/kbp37/dev.txt";

# train mLUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/mluke-base";
poetry run allennlp train examples/relation_classification/configs/transformers_luke.jsonnet -s results/relation_classification/mluke-base --include-package examples -o '{"trainer.cuda_device": 0, "trainer.use_amp": true}'

# train other multilingual models
export TRANSFORMERS_MODEL_NAME="xlm-roberta-base";
poetry run allennlp train examples/relation_classification/configs/transformers.jsonnet  -s results/relation_classification/xlm-roberta-base --include-package examples
```

## Evaluation
```bash
poetry run allennlp evaluate RESULT_SAVE_DIR INPUT_FILE --include-package examples --output-file OUTPUT_FILE 

# example of LUKE
poetry run allennlp evaluate results/relation_classification/luke-base data/tacred/test.json --include-package examples --output-file results/relation_classification/luke-base/metrics_test.json --cuda 0

# example of mLUKE
poetry run allennlp evaluate results/relation_classification/mluke-base data/RELX/Datasets/RELX/RELX_es.txt --include-package examples --output-file results/relation_classification/mluke-base/metrics_relx_es.json --cuda 0
```

## Make Prediction
```bash
poetry run allennlp predict RESULT_SAVE_DIR INPUT_FILE --use-dataset-reader --include-package examples --cuda-device CUDA_DEVICE --output-file OUTPUT_FILE

# example of LUKE
poetry run allennlp predict results/relation_classification/luke-base data/tacred/dev.json --use-dataset-reader --include-package examples --cuda-device 0 --output-file results/relation_classification/luke-base/prediction.json
```

