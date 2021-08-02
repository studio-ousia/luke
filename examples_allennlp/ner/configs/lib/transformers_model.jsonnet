local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

{
    "type": "span_ner",
    "feature_extractor": {
        "type": "token",
        "embedder": {
            "type": "pretrained_transformer",
            "model_name": transformers_model_name
        }
    }
}