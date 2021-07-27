local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

{
    "type": "entity_typing",
    "feature_extractor": {
        "type": "entity",
        "embedder": {
            "type": "transformers-luke",
            "model_name": transformers_model_name,
            "output_embeddings": "entity"
        }
    }
}