local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

{
    "type": "transformers_qa",
    "embedder": {
        "type": "basic",
        "token_embedders": {
            "tokens": {
                "type": "transformers-luke",
                "model_name": transformers_model_name,
                "use_entity_aware_attention": true
            }
        },
    },
}
