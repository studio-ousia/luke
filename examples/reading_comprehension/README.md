# Reading Comprehension
In this code, you can experiment with the task of reading comprehension (extractive QA) with several datasets. Currently, we support the following datasets.

## Datasets 

#####  English 
* [SQuAD](https://aclanthology.org/D16-1264/)

##### Cross-lingual Evaluation
* [XQuAD](https://aclanthology.org/2020.acl-main.421/)
* [MLQA](https://aclanthology.org/2020.acl-main.653/)

##### Download datasets
Download the SQuAD dataset from [here](https://rajpurkar.github.io/SQuAD-explorer).
The XQuAD and MLQA can be obtained through the following commands.
```bash
cd data
git clone https://github.com/deepmind/xquad.git
wget https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip
unzip MLQA_V1.zip
```

## Training
We configure some parameters through environmental variables.  

### SQuAD
```bash
export TRAIN_DATA_PATH="data/SQuAD/train-v1.1.json";
export VALIDATION_DATA_PATH="data/SQuAD/dev-v1.1.json";

# train LUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-large";
poetry run allennlp train examples/reading_comprehension/configs_squad/transformers_luke_with_entity_aware_attention.jsonnet -s results/reading_comprehension/luke-large --include-package examples -o '{"trainer.cuda_device": 0, "trainer.use_amp": true}'

# train mLUKE
export TRANSFORMERS_MODEL_NAME="studio-ousia/mluke-base";
poetry run allennlp train examples/reading_comprehension/configs_squad/transformers_luke.jsonnet -s results/reading_comprehension/mluke-base --include-package examples -o '{"trainer.cuda_device": 0, "trainer.use_amp": true}'

# you can also fine-tune models from the BERT family
export TRANSFORMERS_MODEL_NAME="roberta-base";
poetry run allennlp train examples/reading_comprehension/configs_squad/transformers.jsonnet  -s results/reading_comprehension/roberta-base --include-package examples
```

## Evaluation
To perform cross-lingual evaluation with entity representation, download the files of mention candidates from [here](https://drive.google.com/file/d/12m-mV8sud4F3yXtiVh5QXp3SBPLh_Eje/view?usp=sharing).
The following examples assume the mention candidates are under `data` directory.

```bash
poetry run allennlp evaluate RESULT_SAVE_DIR INPUT_FILE --include-package examples --output-file OUTPUT_FILE 

# example of LUKE
poetry run allennlp evaluate results/reading_comprehension/luke-large data/SQuAD/dev-v1.1.json --include-package examples --output-file results/reading_comprehension/luke-large/metrics_test.json --cuda 0

# example of mLUKE with XQuAD
poetry run python examples/reading_comprehension/evaluate_qa.py results/reading_comprehension/mluke-base data/xquad/xquad.ar.json --mention-candidate-files '{"en-ar": "data/squad_mention_candidates/xquad/dev-v1.1.ar.json"}' --cuda-device 0

# example of mLUKE with MLQA
poetry run python examples/reading_comprehension/evaluate_qa.py results/reading_comprehension/mluke-base data/MLQA_V1/test/test-context-zh-question-zh.json --mention-candidate-files '{"zh-zh": "data/squad_mention_candidates/mlqa/test-context-zh-question-zh.json"}' --cuda-device 0

# example of mLUKE with MLQA in the G-XLT setting (the question and context are in different languages)
poetry run python examples/reading_comprehension/evaluate_qa.py results/reading_comprehension/mluke-base data/MLQA_V1/test/test-context-en-question-zh.json --mention-candidate-files '{"en-en": "data/squad_mention_candidates/mlqa/test-context-en-question-en.json", "en-zh": "data/squad_mention_candidates/mlqa/test-context-en-question-zh.trans.json"}' --cuda-device 0
```
