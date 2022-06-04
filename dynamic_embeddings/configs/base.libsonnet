local luke_model_name = std.extVar("LUKE_MODEL_NAME");

{
    dataset_reader: {
        type: "hyperlink",
        luke_model_name: luke_model_name,
    },
    model: {
        type: "entity_embedding_predictor",
        luke_model_name: luke_model_name,
        freeze_encoder: false
    },
    train_data_path: std.extVar("TRAIN_DATA_PATH"),
    validation_data_path: std.extVar("VALIDATION_DATA_PATH"),
    trainer: {
        cuda_device: -1,
        num_epochs: 3,
        checkpointer: {
            keep_most_recent_by_count: 1
        },
        optimizer: {
            type: "adamw",
            lr: 1e-5,
            betas: [0.9, 0.98],
            eps: 1e-6,
            weight_decay: 0.01,
            parameter_groups: [
                [
                    [
                        "bias",
                        "LayerNorm.weight",
                    ],
                    {
                        weight_decay: 0
                    }
                ]
            ],
        },
        learning_rate_scheduler: {
            type: "custom_linear_with_warmup",
            warmup_ratio: 0.06
        },
        num_gradient_accumulation_steps: 1,
        patience: 2,
        validation_metric: "-loss"
    },
    data_loader: {batch_size: 8, shuffle: true},
    random_seed: 0,
    numpy_seed: 0,
    pytorch_seed: 0
}
