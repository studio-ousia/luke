local seed = std.parseInt(std.extVar("SEED"));
local train_data_path = std.extVar("TRAIN_DATA_PATH");
local validation_data_path = std.extVar("VALIDATION_DATA_PATH");

local lr =  2e-5;
local batch_size = 2;
local accumulation_steps = 16;
local num_epochs = 2;

{
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "data_loader": {
        "batch_size": batch_size, "shuffle": true
    },
     "validation_data_loader": {
        "batch_size": batch_size, "shuffle": false
    },
    "trainer": {
        "num_epochs": num_epochs,
        "patience": 3,
        "cuda_device": -1,
        "grad_norm": 5.0,
        "num_gradient_accumulation_steps": accumulation_steps,
        "checkpointer": {
            "keep_most_recent_by_count": 1
        },
        "validation_metric": "-loss",
        "optimizer": {
            "type": "adamw",
            "lr": lr,
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
    },
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed
}
