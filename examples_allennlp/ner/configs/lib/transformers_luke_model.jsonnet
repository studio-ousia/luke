local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

{
    "type": "span_ner",
    "feature_extractor": {
        "type": "token-entity",
        "embedder": {
            "type": "transformers-luke",
            "model_name": transformers_model_name,
            "output_embeddings": "tokens+entity"
        }
    }
}