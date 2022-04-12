local base = import "lib/base.libsonnet";
local model = import "lib/transformers_model_luke_with_entity_aware_attention.jsonnet";

base + {
    "model": model,
    "dataset_reader": base["dataset_reader"] + {"use_entity_feature": true},
}