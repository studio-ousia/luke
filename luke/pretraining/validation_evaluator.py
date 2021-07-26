from allennlp.common.registrable import Registrable


class ValidationEvaluator(Registrable):
    def __call__(self, model) -> float:
        raise NotImplementedError
