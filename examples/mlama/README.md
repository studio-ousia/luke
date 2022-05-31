In this code, you can experiment with the task of a multilingual cloze prompt task.  
Currently, we support the following datasets.

* [mLAMA](https://aclanthology.org/2021.eacl-main.284/)

# Download datasets
```bash
cd data
git clone https://github.com/norakassner/mlama.git
cd mlama/data
wget http://cistern.cis.lmu.de/mlama/mlama1.1.zip
unzip mlama1.1.zip
rm mlama1.1.zip
```

## Evaluation
```bash
poetry run python examples/mlama/evaluate.py data/mlama/data en bert-base-cased
poetry run python examples/mlama/evaluate.py data/mlama/data en luke
```
