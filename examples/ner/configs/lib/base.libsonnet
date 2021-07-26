local seed = std.parseInt(std.extVar("SEED"));
local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local tokenizer = {"type": "pretrained_transformer", "model_name": transformers_model_name,
                    "add_special_tokens": false,  "tokenizer_kwargs": {"add_prefix_space": true}};
local token_indexers = {
            "tokens": {"type": "pretrained_transformer", "model_name": transformers_model_name}
    };


{
    "dataset_reader": {
        "type": "conll_span",
        "tokenizer": tokenizer,
        "token_indexers": token_indexers,
        "encoding": "utf-8",
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
    "trainer": {
        "cuda_device": -1,
        "grad_norm": 5,
        "num_epochs": 5,
        "checkpointer": {
            "keep_most_recent_by_count": 0
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-5,
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
        "num_gradient_accumulation_steps": 4,
        "patience": 3,
        "validation_metric": "+f1"
    },
    "data_loader": {"batch_size": 2, "shuffle": true},
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed
}