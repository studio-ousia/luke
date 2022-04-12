local base = import "lib/base.libsonnet";
local model = import "lib/transformers_model_luke_with_entity_aware_attention.jsonnet";
local dataset_reader = import "lib/dataset_reader_with_entity.jsonnet";

base + {"model": model, "dataset_reader": dataset_reader}
