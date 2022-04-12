local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local base = import "lib/base.libsonnet";
local model = import "lib/transformers_model_luke.jsonnet";

base + {
    "model": model,
    "dataset_reader": base["dataset_reader"] + {"use_entity_feature": true},
}