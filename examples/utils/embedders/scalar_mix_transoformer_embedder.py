from allennlp.modules.token_embedders import TokenEmbedder, PretrainedTransformerEmbedder
from allennlp.modules.scalar_mix import ScalarMix


@TokenEmbedder.register("intermediate_pretrained_transformer")
class IntermediatePretrainedTransformerEmbedder(PretrainedTransformerEmbedder):
    def __init__(self, layer_index: int, **kwargs) -> None:
        super().__init__(**kwargs, last_layer_only=False)

        initial_scalar_parameters = [-1e9 for _ in range(self.config.num_hidden_layers)]
        initial_scalar_parameters[layer_index] = 0

        self._scalar_mix = ScalarMix(
            self.config.num_hidden_layers,
            initial_scalar_parameters=initial_scalar_parameters,
            trainable=False,
            do_layer_norm=False,
        )
