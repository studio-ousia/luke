local seed = std.parseInt(std.extVar("SEED"));
local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local extra_tokens = ["<ent>"];

local tokenizer = {"type": "pretrained_transformer",
                   "model_name": transformers_model_name,
                   "add_special_tokens": true,
                   "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}};
local token_indexers = {
            "tokens": {"type": "pretrained_transformer", "model_name": transformers_model_name,
                       "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}
    }};

{
    "dataset_reader": {
        "type": "entity_typing",
        "tokenizer": tokenizer,
        "token_indexers": token_indexers
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
    "trainer": {
        "cuda_device": -1,
        "num_epochs": 3,
        "checkpointer": {
            "keep_most_recent_by_count": 0
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-5,
            "betas": [0.9, 0.98],
            "eps": 1e-6,
            "weight_decay": 0.01,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.weight",
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
        },
        "learning_rate_scheduler": {
            "type": "custom_linear_with_warmup",
            "warmup_ratio": 0.06
        },
        "num_gradient_accumulation_steps": 1,
        "patience": 3,
        "validation_metric": "+micro_fscore"
    },
    "data_loader": {"batch_size": 4, "shuffle": true},
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed
}