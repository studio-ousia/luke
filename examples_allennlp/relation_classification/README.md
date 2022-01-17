# Relation Classification
In this code, you can experiment with the task of relation classification with several datasets. Currently, we support the following datasets.


#####  English 
* [TACRED](https://www.aclweb.org/anthology/D17-1004/)

For the TACRED dataset, you need to access through [LDC](https://catalog.ldc.upenn.edu/LDC2018T24).

# Reproduce the checkpoint result
```bash
# We assume you are in this directory (examples_allennlp/relation_classification). 
poetry run python evaluate_transformers_checkpoint.py TEST_DATA_PATH
# expected results:
# {'accuracy': 0.8887742601070346, 'macro_fscore': 0.5886632942601121, 'micro_fscore': 0.7267450297489478}.
```

# Training
We configure some parameters through environmental variables.  
Note that you need provide what dataset you are dealing with through `TASK`.
```bash
export SEED=0;
export TRAIN_DATA_PATH="data/tacred/train.json";
export VALIDATION_DATA_PATH="data/tacred/dev.json";

# train LUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-base";
poetry run allennlp train examples_allennlp/relation_classification/configs/transformers_luke.jsonnet -s results/relation_classification/luke-base --include-package examples_allennlp -o '{"trainer": {"cuda_device": 0}}'

# you can also fine-tune models from the BERT family
export TRANSFORMERS_MODEL_NAME="roberta-base";
poetry run allennlp train examples_allennlp/relation_classification/configs/transformers.jsonnet  -s results/relation_classification/roberta-base --include-package examples_allennlp
```

# Evaluation
```bash
poetry run allennlp evaluate RESULT_SAVE_DIR INPUT_FILE --include-package examples_allennlp --output-file OUTPUT_FILE 

# example for LUKE
poetry run allennlp evaluate results/relation_classification/luke-base data/tacred/test.json --include-package examples_allennlp --output-file results/relation_classification/luke-base/metrics_test.json --cuda 0
```

# Make Prediction
```bash
poetry run allennlp predict RESULT_SAVE_DIR INPUT_FILE --use-dataset-reader --include-package examples_allennlp --cuda-device CUDA_DEVICE --output-file OUTPUT_FILE

# example for LUKE
poetry run allennlp predict results/relation_classification/luke-base data/tacred/dev.json --use-dataset-reader --include-package examples_allennlp --cuda-device 0 --output-file results/relation_classification/luke-base/prediction.json
```

