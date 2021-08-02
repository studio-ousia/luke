import torch


def get_span_max_length(span: torch.LongTensor) -> int:
    return (span[:, 1] - span[:, 0] + 1).max().item()


def span_to_position_ids(span: torch.LongTensor, max_length: int = None) -> torch.LongTensor:
    batch_size = span.size(0)
    max_length = max_length or get_span_max_length(span)
    position_ids = span.new_full((batch_size, max_length), fill_value=-1)

    for i, (start, end) in enumerate(span):
        positions = torch.arange(start, end + 1)
        position_ids[i, : len(positions)] = positions
    return position_ids


def span_pooling(token_embeddings: torch.Tensor, span: torch.LongTensor) -> torch.Tensor:
    """
    Parameters
    ----------
    token_embeddings: (batch_size, sequence_length, feature_size)
    span: (batch_size, 2)

    Returns
    ---------
    pooled_embeddings: (batch_size, feature_size)
    """
    pooled_embeddings = []
    for token_emb, (start, end) in zip(token_embeddings, span):
        # The span indices are as follows and we only pool among the word positions.
        # start, ...    , end
        # <e>,   w0, w1,  </e>
        pooled_emb, _ = token_emb[start + 1 : end].max(dim=0)
        pooled_embeddings.append(pooled_emb)
    return torch.stack(pooled_embeddings)


def extract_span_start(token_embeddings: torch.Tensor, span: torch.LongTensor) -> torch.Tensor:
    entity_start_position = span[:, 0]
    batch_size, _, embedding_size = token_embeddings.size()
    range_tensor = torch.arange(batch_size, device=token_embeddings.device)
    start_embeddings = token_embeddings[range_tensor, entity_start_position]
    return start_embeddings
