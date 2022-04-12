local base = import "lib/base.libsonnet";
local model = import "lib/transformers_model.jsonnet";

base + {"model": model}