local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

{
    "type": "entity_typing",
    "feature_extractor": {
        "type": "token",
        "embedder": {
            "type": "pretrained_transformer",
            "model_name": transformers_model_name,
            "tokenizer_kwargs": {"additional_special_tokens": ["<ent>"]}
        },
        "feature_type": "entity_start"
    }
}