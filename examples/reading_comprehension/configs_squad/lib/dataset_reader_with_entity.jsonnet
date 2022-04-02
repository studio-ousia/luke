local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

{
    "type": "transformers_squad",
    "transformer_model_name": transformers_model_name,
    "skip_impossible_questions": false,
    "wiki_entity_linker": {
        "type": "json",
        "mention_candidate_json_file_paths": {
            "en-en": "examples/reading_comprehension/mention_candidates/squad/train-dev-v1.1.json"
        },
        "entity_vocab_path": transformers_model_name,
    }
}
