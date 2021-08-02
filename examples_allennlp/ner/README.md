
In this code, you can experiment with the task of named entity recognition (NER).  
Currently, we support the following datasets.


#####  English
* [CoNLL-2003 shared task](https://aclanthology.org/W03-0419/)

To obtain the dataset, follow the instruction in the website above.
We assume that the files follow the CoNLL-2003 format.

# Reproduce the checkpoint result
```bash
# We assume you are in this directory (examples/ner). 
poetry run python evaluate_transformers_checkpoint.py TEST_DATA_PATH
# expected results:
# {'f1': 0.9461946902654867, 'precision': 0.945859872611465, 'recall': 0.9465297450424929}.
```

You may find the result a little higher than the original paper of LUKE.  
This is due to a minor difference in the data preprocessing in the evaluation code.

# Training
We configure some parameters through environmental variables.
```bash
# We assume you are in the root directory of luke. 

export SEED=0;
export TRAIN_DATA_PATH="data/ner/train.conllu";
export VALIDATION_DATA_PATH="data/ner/dev.conllu";

# train LUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-base";
poetry run allennlp train examples/ner/configs/transformers_luke.jsonnet -s results/ner/luke-base --include-package examples -o '{"trainer": {"cuda_device": 0}}'

# you can also fine-tune models from the BERT family
export TRANSFORMERS_MODEL_NAME="roberta-base";
poetry run allennlp train examples/ner/configs/transformers.jsonnet  -s results/ner/roberta-base --include-package examples
```

# Evaluation
```bash
poetry run allennlp evaluate RESULT_SAVE_DIR INPUT_FILE --include-package examples --output-file OUTPUT_FILE 

# example for LUKE
poetry run allennlp evaluate results/ner/luke-base data/ner/test.conllu --include-package examples --output-file results/ner/luke-base/metrics_test.json --cuda 0
```