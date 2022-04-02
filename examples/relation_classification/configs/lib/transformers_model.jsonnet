local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

{
    "type": "relation_classifier",
    "feature_extractor": {
        "type": "token",
        "embedder": {
            "type": "pretrained_transformer",
            "model_name": transformers_model_name,
            "tokenizer_kwargs": {"additional_special_tokens": ["<ent>", "<ent2>"]}
        },
        "feature_type": "entity_start"
    }
}