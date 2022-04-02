In this code, you can experiment with the task of entity typing.  
Currently, we support the following datasets.


#####  English 
* [Open Entity](https://www.aclweb.org/anthology/P18-1009/)

# Download datasets
```bash
cd data
wget http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz
tar -zxvf ultrafine_acl18.tar.gz
mv release ultrafine_acl18
```
We only use the manually annotated data under `ultrafine_acl18/crowd` for training and evaluation.

## Reproduce the checkpoint result
```bash
# Reproduce the result of studio-ousia/luke-large-finetuned-open-entity.
poetry run python examples/entity_typing/evaluate_transformers_checkpoint.py data/ultrafine_acl18/crowd/test.json
# expected results:
# {'micro_precision': 0.7997806072235107, 'micro_recall': 0.7657563090324402, 'micro_fscore': 0.7823987007141113}.
```

# Training
We configure some parameters through environmental variables.
```bash
export TRAIN_DATA_PATH="data/ultrafine_acl18/crowd/train.json";
export VALIDATION_DATA_PATH="data/ultrafine_acl18/crowd/dev.json";

# train LUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-base";
poetry run allennlp train examples/entity_typing/configs/transformers_luke_with_entity_aware_attention.jsonnet -s results/entity_typing/luke-base --include-package examples -o '{"trainer.cuda_device": 0, "trainer.use_amp": true}'

# you can also fine-tune models from the BERT family
export TRANSFORMERS_MODEL_NAME="roberta-base";
poetry run allennlp train examples/entity_typing/configs/transformers.jsonnet  -s results/entity_typing/roberta-base --include-package examples
```

# Evaluation
```bash
poetry run allennlp evaluate RESULT_SAVE_DIR INPUT_FILE --include-package examples --output-file OUTPUT_FILE 

# example of LUKE
poetry run allennlp evaluate results/entity_typing/luke-base data/ultrafine_acl18/crowd/test.json --include-package examples --output-file results/entity_typing/luke-base/metrics_test.json --cuda 0
```

# Make Prediction
```bash
poetry run allennlp predict RESULT_SAVE_DIR INPUT_FILE --use-dataset-reader --include-package examples --cuda-device CUDA_DEVICE --output-file OUTPUT_FILE

# example of LUKE
poetry run allennlp predict results/entity_typing/luke-base data/ultrafine_acl18/crowd/dev.json --use-dataset-reader --include-package examples --cuda-device 0 --output-file results/entity_typing/luke-base/prediction.json
```

