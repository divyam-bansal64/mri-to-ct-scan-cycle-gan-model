# config/hyperparameter_grid_confirm.py

HYPERPARAM_GRID = [
    {
        "exp_name": "confirm_run_012",
        "lr_G": 1e-4,
        "lr_D": 1e-4,
        "lambda_cycle": 5.0,
        "lambda_identity": 2.5,
        "n_blocks": 4,
        "n_layers_D": 3,
        "use_spect": True
    },
    {
        "exp_name": "confirm_run_018",
        "lr_G": 2e-4,
        "lr_D": 1e-4,
        "lambda_cycle": 5.0,
        "lambda_identity": 2.5,
        "n_blocks": 6,
        "n_layers_D": 3,
        "use_spect": False
    },
    {
        "exp_name": "confirm_run_004",
        "lr_G": 2e-4,
        "lr_D": 1e-4,
        "lambda_cycle": 5.0,
        "lambda_identity": 2.5,
        "n_blocks": 4,
        "n_layers_D": 3,
        "use_spect": True
    },
    {
        "exp_name": "confirm_run_002",
        "lr_G": 2e-4,
        "lr_D": 1e-4,
        "lambda_cycle": 5.0,
        "lambda_identity": 2.5,
        "n_blocks": 4,
        "n_layers_D": 3,
        "use_spect": False
    }
]
