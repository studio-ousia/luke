from typing import Optional
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
from transformers import AutoTokenizer

from examples.relation_classification.luke_for_entity_pair_classification import LukeForEntityPairClassification
from examples.relation_classification.reader import RelationClassificationReader
from examples.utils.util import ENT, ENT2

model_config_mapping = {
    "studio-ousia/luke-large-finetuned-tacred": "examples/relation_classification/configs/lib/transformers_model_luke_with_entity_aware_attention.jsonnet",
    "studio-ousia/mluke-large-lite-finetuned-kbp37": "examples/relation_classification/configs/lib/transformers_model_luke.jsonnet",
}

tokenizer_name_mapping = {
    "studio-ousia/luke-large-finetuned-tacred": "studio-ousia/luke-large",
    "studio-ousia/mluke-large-lite-finetuned-kbp37": "studio-ousia/mluke-large-lite",
}


@click.command()
@click.argument("dataset", default="tacred")
@click.argument("data-path", default="data/tacred/test.json")
@click.argument("checkpoint-model-name", type=str, default="studio-ousia/luke-large-finetuned-tacred")
@click.option("--checkpoint-tokenizer-name", type=str)
@click.option("--model-config-path", type=click.Path(exists=True))
@click.option("--batch-size", type=int, default=32)
@click.option("--cuda-device", type=int, default=0)
@click.option("--result-save-path", type=click.Path(exists=False), default=None)
def evaluate_transformers_checkpoint(
    dataset: str,
    data_path: str,
    checkpoint_model_name: str,
    checkpoint_tokenizer_name: Optional[str],
    model_config_path: Optional[str],
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

    checkpoint_tokenizer_name = checkpoint_tokenizer_name or tokenizer_name_mapping.get(checkpoint_model_name)
    if checkpoint_tokenizer_name is None:
        raise ValueError("You need to specify which tokenizer to use with a new checkpoint.")
    print(f"Use the tokenizer: {checkpoint_tokenizer_name}")

    tokenizer_kwargs = {"additional_special_tokens": [ENT, ENT2]}
    reader = RelationClassificationReader(
        dataset=dataset,
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

    transformers_tokenizer = AutoTokenizer.from_pretrained(checkpoint_model_name)
    transformers_model = LukeForEntityPairClassification.from_pretrained(checkpoint_model_name)

    vocab = Vocabulary()
    vocab.add_transformer_vocab(transformers_tokenizer, "tokens")
    num_labels = len(transformers_model.config.id2label)
    labels = [transformers_model.config.id2label[i] for i in range(num_labels)]
    vocab.add_tokens_to_namespace(labels, namespace="labels")

    # read model
    model_config_path = model_config_path or model_config_mapping.get(checkpoint_model_name)
    if model_config_path is None:
        raise ValueError("You need to specify which model config file to use with a new checkpoint.")
    print(f"Use the model config: {model_config_path}")

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
