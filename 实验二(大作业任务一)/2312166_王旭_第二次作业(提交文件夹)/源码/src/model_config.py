from __future__ import annotations


BIAS_BASELINE_CONFIGURATION = {
    "regularization": 10.0,
    "epochs": 10,
}

ITEM_CF_CONFIGURATION = {
    "neighbors": 20,
    "minimum_common_users": 2,
    "similarity_shrinkage": 10.0,
    "similarity_cache_size": 250_000,
}

ENSEMBLE_ITEM_CF_WEIGHT = 0.6551032963199183


def matrix_factorization_configuration(seed: int) -> dict[str, float | int]:
    return {
        "factors": 5,
        "epochs": 8,
        "learning_rate": 0.002,
        "learning_rate_decay": 0.95,
        "factor_regularization": 0.5,
        "bias_regularization": 0.1,
        "seed": seed,
    }
