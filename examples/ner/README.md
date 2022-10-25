# Named Entity Recognition (NER)
In this code, you can experiment with the task of named entity recognition (NER).  

## Datasets
Currently, we support the following datasets.
* [CoNLL-2003 shared task](https://aclanthology.org/W03-0419/) (English and German)
* [CoNLL-2002 shared task](https://aclanthology.org/W02-2024/) (Dutch and Spanish)  
We assume that the data files follow the CoNLL-2003 format.

## Reproduce the checkpoint results
### LUKE-large
```bash
# Reproduce the result of studio-ousia/luke-large-finetuned-conll-2003.
python examples/ner/evaluate_transformers_checkpoint.py data/ner_conll/en/test.txt studio-ousia/luke-large-finetuned-conll-2003 --cuda-device 0
# Expected results (you may observe slightly different results due to environmental differences):
# {'f1': 0.946, 'precision': 0.945, 'recall': 0.946}.
```
You may find the result a little higher than [the original paper of LUKE](https://arxiv.org/abs/2010.01057). This is due to a minor difference in the data preprocessing in the evaluation code.

### mLUKE-large
```bash
# Reproduce the result of studio-ousia/mluke-large-lite-finetuned-conll-2003
python examples/ner/evaluate_transformers_checkpoint.py data/ner_conll/de/deu.testb.bio  studio-ousia/mluke-large-lite-finetuned-conll-2003 --cuda-device 0

poetry run python examples/ner/evaluate_transformers_checkpoint.py data/ner_conll/en/test.txt  studio-ousia/mluke-large-lite-finetuned-conll-2003 --cuda-device 0



# When the input file has a file encoding different from utf-8, you should specify it with --file-encoding.
python examples/ner/evaluate_transformers_checkpoint.py data/ner_conll/es/esp.testb studio-ousia/mluke-large-lite-finetuned-conll-2003 --cuda-device 0 --file-encoding ISO-8859-1

# Expected results for each language.
# {"en": 94.1, "de": 78.8 "nl": 82.4 "es": 82.2", "average": 84.4}  
```
The model is not the exact same model as the one reported in [the paper](https://arxiv.org/abs/2110.08151). You will see some deviation in the language-wise scores from the paper, but the overall performance is the same.

## Training
We configure some parameters through environmental variables.
```bash
# We assume you are in the root directory of luke. 

export TRAIN_DATA_PATH="data/ner_conll/en/train.txt";
export VALIDATION_DATA_PATH="data/ner_conll/en/valid.txt";

# train LUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-large";
allennlp train examples/ner/configs/transformers_luke_with_entity_aware_attention.jsonnet -s results/ner/luke-large --include-package examples -o '{"trainer.cuda_device": 0, "trainer.use_amp": true}'

# train mLUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/mluke-base";
allennlp train examples/ner/configs/transformers_luke.jsonnet -s results/ner/mluke-base --include-package examples -o '{"trainer.cuda_device": 0, "trainer.use_amp": true}'

# you can also fine-tune models from the BERT family
export TRANSFORMERS_MODEL_NAME="roberta-base";
allennlp train examples/ner/configs/transformers.jsonnet  -s results/ner/roberta-base --include-package examples
```

## Evaluation
```bash
allennlp evaluate RESULT_SAVE_DIR INPUT_FILE --include-package examples --output-file OUTPUT_FILE 

# example of LUKE
allennlp evaluate results/ner/luke-large /data/ner_conll/en/test.txt --include-package examples --output-file results/ner/luke-large/metrics_test.json --cuda 0

# example of mLUKE (cross-lingual transfer)
allennlp evaluate results/ner/mluke-base data/ner_conll/de/deu.testb.bio --include-package examples --output-file results/ner/mluke-base/metrics_de_test.json --cuda 0
```


## Conversion from Allennlp to HuggingFace Model
```bash
python examples/ner/convert_allennlp_to_huggingface_model.py SERIALIZATION-DIR SAVE-DIR
```
