In this code, you can experiment with the task of entity typing.  
Currently, we support the following datasets.


#####  English 
* [Open Entity](https://www.aclweb.org/anthology/P18-1009/)

# Download datasets
```bash
cd data
wget http://nlp.cs.washington.edu/entity_type/data/release.tar.gz
tar -zxvf ultrafine_acl18.tar.gz
```

We only use the manually annotated data under `release/crowd` for training and evaluation.

# Reproduce the checkpoint result
```bash
# We assume you are in this directory (examples_allennlp/entity_typing). 
poetry run python evaluate_transformers_checkpoint.py TEST_DATA_PATH
# expected results:
# {'micro_precision': 0.7997806072235107, 'micro_recall': 0.7657563090324402, 'micro_fscore': 0.7823987007141113}.
```

# Training
We configure some parameters through environmental variables.
```bash
export SEED=0;
export TRAIN_DATA_PATH="data/release/crowd/train.json";
export VALIDATION_DATA_PATH="data/release/crowd/dev.json";

# train LUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-base";
poetry run allennlp train examples_allennlp/entity_typing/configs/transformers_luke.jsonnet -s results/entity_typing/luke-base --include-package examples_allennlp -o '{"trainer": {"cuda_device": 0}}'

# you can also fine-tune models from the BERT family
export TRANSFORMERS_MODEL_NAME="roberta-base";
poetry run allennlp train examples_allennlp/entity_typing/configs/transformers.jsonnet  -s results/entity_typing/roberta-base --include-package examples_allennlp
```

# Evaluation
```bash
poetry run allennlp evaluate RESULT_SAVE_DIR INPUT_FILE --include-package examples_allennlp --output-file OUTPUT_FILE 

# example for LUKE
poetry run allennlp evaluate results/entity_typing/luke-base data/release/crowd/test.json --include-package examples_allennlp --output-file results/entity_typing/luke-base/metrics_test.json --cuda 0
```

# Make Prediction
```bash
poetry run allennlp predict RESULT_SAVE_DIR INPUT_FILE --use-dataset-reader --include-package examples_allennlp --cuda-device CUDA_DEVICE --output-file OUTPUT_FILE

# example for LUKE
poetry run allennlp predict results/entity_typing/luke-base data/release/crowd/dev.json --use-dataset-reader --include-package examples_allennlp --cuda-device 0 --output-file results/entity_typing/luke-base/prediction.json
```

