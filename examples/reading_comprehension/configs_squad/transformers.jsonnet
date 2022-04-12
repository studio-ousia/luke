local base = import "lib/base.libsonnet";
local model = import "lib/transformers_model.jsonnet";
local dataset_reader = import "lib/dataset_reader.jsonnet";

base + {"model": model, "dataset_reader": dataset_reader}
