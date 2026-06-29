# configs/hyperparameter_grid_full_model.py
from itertools import product

# Top 7 stable generator configurations (based on earlier analysis)
GENERATOR_CONFIGS = [
    {"lr_G": 2e-4, "lambda_cycle": 5.0, "lambda_identity": 2.5, "n_blocks": 4},
    {"lr_G": 1e-4, "lambda_cycle": 5.0, "lambda_identity": 5.0, "n_blocks": 4},
    {"lr_G": 1e-4, "lambda_cycle": 5.0, "lambda_identity": 2.5, "n_blocks": 4},
    {"lr_G": 1e-4, "lambda_cycle": 5.0, "lambda_identity": 5.0, "n_blocks": 6},
    {"lr_G": 2e-4, "lambda_cycle": 5.0, "lambda_identity": 2.5, "n_blocks": 6},
    {"lr_G": 2e-4, "lambda_cycle": 10.0, "lambda_identity": 2.5, "n_blocks": 4},
    {"lr_G": 1e-4, "lambda_cycle": 10.0, "lambda_identity": 5.0, "n_blocks": 4},
]

# Add discriminator parameter variations
DISCRIMINATOR_CONFIGS = [
    {"lr_D": 2e-4, "n_layers_D": 4, "use_spect": False},
    {"lr_D": 1e-4, "n_layers_D": 3, "use_spect": False},
    {"lr_D": 2e-4, "n_layers_D": 4, "use_spect": True},
    {"lr_D": 1e-4, "n_layers_D": 3, "use_spect": True},
]

# Combine generator + discriminator configs into one grid
HYPERPARAM_GRID = []
for g_cfg, d_cfg in product(GENERATOR_CONFIGS, DISCRIMINATOR_CONFIGS):
    combined = {}
    combined.update(g_cfg)
    combined.update(d_cfg)
    HYPERPARAM_GRID.append(combined)
