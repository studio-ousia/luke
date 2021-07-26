import torch
from allennlp.data import TextFieldTensors
from allennlp.models import Model


class Seq2VecEncoder(Model):
    def get_output_dim(self) -> int:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def forward(self, tokens: TextFieldTensors) -> torch.Tensor:
        """
        Parameters :
        embedding_sequence : torch.tensor (batch_size, sequence_length, embedding_size)
            A batch of embedded sequence inputs.
        mask : torch.LongTensor (batch, sequence_length)
            A mask with 0 where the tokens are padding, and 1 otherwise.

        Returns :
        encoder_output : {
            "vector": torch.Tensor (batch_size, output_size)
        }
        """
        raise NotImplementedError
