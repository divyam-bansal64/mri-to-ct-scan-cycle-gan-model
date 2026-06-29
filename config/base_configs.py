# configs/base_configs.py

CONFIG = {
    "image_size": 128,          # keep smaller size for local testing
    "input_nc": 3,
    "output_nc": 3,
    "ngf": 64,
    "ndf": 64,
    "n_blocks": 6,
    "n_layers_D": 4,
    "batch_size": 1,
    "epochs": 30,               # increased from 10 → 30
    "lr_G": 2e-4,
    "lr_D": 2e-4,
    "betas": (0.5, 0.999),
    "lambda_cycle": 10.0,
    "lambda_identity": 5.0,
    "lambda_gan": 1.0,
    "use_spect": False,         # can be overridden by specific configs
    "device": "cuda",
    "save_interval": 10,
    "limit": 50,                # use subset for speed
}
