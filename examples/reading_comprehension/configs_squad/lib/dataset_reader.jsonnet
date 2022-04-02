local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

{"type": "transformers_squad", "transformer_model_name": transformers_model_name, "skip_impossible_questions": false}
