local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

{
    "type": "transformers_qa",
    "embedder": {
        "type": "basic",
        "token_embedders": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformers_model_name,
            }
        },
    },
}
