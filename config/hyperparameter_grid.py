# configs/hyperparameter_grid.py
HYPERPARAM_GRID = {
    "lr_G": [2e-4, 1e-4],
    "lambda_cycle": [5.0, 10.0, 15.0],
    "lambda_identity": [2.5, 5.0],
    "n_blocks": [4, 6],
    "image_size": [128]   # test small for speed, later change to 256
}
