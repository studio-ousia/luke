import json

import click
import torch
import tqdm
from allennlp.common import Params
from allennlp.common.util import import_module_and_submodules
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.nn import util as nn_util
from transformers import LukeForEntityClassification, LukeTokenizer

from examples.entity_typing.reader import EntityTypingReader
from examples.utils.util import ENT


@click.command()
@click.argument("data-path", default="data/ultrafine_acl18/crowd/test.json")
@click.argument(
    "model-config-path", type=click.Path(exists=True), default="examples/entity_typing/configs/lib/transformers_model_luke_with_entity_aware_attention.jsonnet",
)
@click.argument("checkpoint-model-name", type=str, default="studio-ousia/luke-large-finetuned-open-entity")
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
    import_module_and_submodules("examples")

    tokenizer_kwargs = {"additional_special_tokens": [ENT]}
    reader = EntityTypingReader(
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
    transformers_model = LukeForEntityClassification.from_pretrained(checkpoint_model_name)

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
