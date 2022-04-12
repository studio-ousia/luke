# Fine-tuning for downstream tasks 

We use the [allennlp](https://github.com/allenai/allennlp) library as the backbone of our fine-tuning code (except for `entity_disambiguation` and `legacy`). When you want to modify the code, we recommend referring to [the tutorial of allennlp](https://guide.allennlp.org/) to get a basic understanding of the library. 

For further details and examples, see the `README.md` under the directories of each task.

## On the hyper-parameter tuning
The training commands described on `README.md` under each directory are just examples, not necessarily reproducing the results on the papers ([LUKE](https://arxiv.org/abs/2010.01057) or [mLUKE](https://arxiv.org/abs/2110.08151)) with a single run.

For the `allennlp train` command, you can search hyper-parameters by using the `--overrides, -o` option, which looks like this.
```bash
poetry run allennlp train CONFIG_PATH -s SERIALIZATION_DIR --include-package examples --overrides `{"data_loader.batch_size": 8, "trainer.optimizer.lr": 2e-5, "random_seed": 42, "numpy_seed": 42, "pytorch_seed": 42}`
```
With a decent amount of grid search, you should be able to see performance comparable to the original paper.