import click
import json
import tqdm
import torch


from allennlp.common import Params
from allennlp.common.util import import_module_and_submodules
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models import Model
from allennlp.nn import util as nn_util

from transformers import LukeTokenizer, LukeForEntityPairClassification

from examples_allennlp.relation_classification.reader import RelationClassificationReader
from examples_allennlp.utils.util import ENT, ENT2


@click.command()
@click.argument("data-path")
@click.argument(
    "model-config-path", type=click.Path(exists=True), default="configs/lib/transformers_luke_model.jsonnet",
)
@click.argument("checkpoint-model-name", type=str, default="studio-ousia/luke-large-finetuned-tacred")
@click.argument("checkpoint-tokenizer-name", type=str, default="studio-ousia/luke-large")
@click.option("--batch-size", type=int, default=32)
@click.option("--cuda-device", type=int, default=0)
@click.option("--result-save-path", type=click.Path(exists=False), default=None)
def evaluate_transformers_checkpoint(
    data_path: str,
    model_config_path: str,
    checkpoint_model_name: str,
    checkpoint_tokenizer_name: str,
    batch_size: int,
    cuda_device: int,
    result_save_path: str,
):
    """
    Expected results for ``test.json`` from the TACRED dataset:
    {'accuracy': 0.8887742601070346, 'macro_fscore': 0.5886632942601121, 'micro_fscore': 0.7267450297489478}.

    Parameters
    ----------
    data_path : str
        Data path to the input file.
    model_config_path : str
        A config file that defines the model architecture to evaluate.
    checkpoint_model_name : str
        The name of the checkpoint in Hugging Face Model Hub.
    checkpoint_tokenizer_name : str
        This should be the name of the base pre-training model because sometimes
        the tokenizer of downstream task is not compatible with allennlp.
    batch_size : int
    cuda_device : int
    result_save_path : str
    """
    import_module_and_submodules("examples_allennlp")

    tokenizer_kwargs = {"additional_special_tokens": [ENT, ENT2]}
    reader = RelationClassificationReader(
        tokenizer=PretrainedTransformerTokenizer(
            model_name=checkpoint_tokenizer_name, add_special_tokens=True, tokenizer_kwargs=tokenizer_kwargs
        ),
        token_indexers={
            "tokens": PretrainedTransformerIndexer(
                model_name=checkpoint_tokenizer_name, tokenizer_kwargs=tokenizer_kwargs
            )
        },
        use_entity_feature=True,
    )

    transformers_tokenizer = LukeTokenizer.from_pretrained(checkpoint_model_name)
    transformers_model = LukeForEntityPairClassification.from_pretrained(checkpoint_model_name)

    vocab = Vocabulary()
    vocab.add_transformer_vocab(transformers_tokenizer, "tokens")
    num_labels = len(transformers_model.config.id2label)
    labels = [transformers_model.config.id2label[i] for i in range(num_labels)]
    vocab.add_tokens_to_namespace(labels, namespace="labels")

    # read model
    params = Params.from_file(model_config_path, ext_vars={"TRANSFORMERS_MODEL_NAME": checkpoint_model_name})
    model = Model.from_params(params, vocab=vocab)
    model.classifier = transformers_model.classifier
    model.eval()

    # set the GPU device to use
    if cuda_device < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cuda_device}")
    model = model.to(device)

    loader = MultiProcessDataLoader(reader, data_path, batch_size=batch_size, shuffle=False)
    loader.index_with(model.vocab)
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            batch = nn_util.move_to_device(batch, device)
            output_dict = model(**batch)

    metrics = model.get_metrics(reset=True)
    print(metrics)
    if result_save_path is not None:
        with open(result_save_path, "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    evaluate_transformers_checkpoint()
