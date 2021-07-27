local base = import "lib/base.libsonnet";
local model = import "lib/transformers_luke_model.jsonnet";


base + {
    "dataset_reader": base["dataset_reader"] + {"use_entity_feature": true},
    "model": model
}