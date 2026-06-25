from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

from src.data_io import load_ratings, split_ratings_by_user
from src.ensemble import blend_predictions, optimal_blend_weight
from src.item_cf import ItemItemCFModel
from src.matrix_factorization import BiasedMatrixFactorizationModel
from src.metrics import rmse
from src.model_config import ITEM_CF_CONFIGURATION
from src.model_config import ENSEMBLE_ITEM_CF_WEIGHT
from src.model_config import matrix_factorization_configuration


def fit_and_predict(model, training, validation) -> tuple[list[float], float, float]:
    training_started = time.perf_counter()
    model.fit(training)
    training_seconds = time.perf_counter() - training_started
    prediction_started = time.perf_counter()
    predictions = [
        model.predict(user_id, item_id) for user_id, item_id, _ in validation
    ]
    prediction_seconds = time.perf_counter() - prediction_started
    return predictions, training_seconds, prediction_seconds


def main() -> None:
    seed = 2026
    ratings = load_ratings(Path("data/train.txt"))
    training, validation = split_ratings_by_user(ratings, 0.2, seed)
    actual = [score for _, _, score in validation]

    item_cf = ItemItemCFModel(10.0, 100.0, **ITEM_CF_CONFIGURATION)
    matrix_factorization = BiasedMatrixFactorizationModel(
        10.0, 100.0, **matrix_factorization_configuration(seed)
    )

    tracemalloc.start()
    print("Fitting and predicting with Item-CF...", flush=True)
    item_predictions, item_training, item_prediction = fit_and_predict(
        item_cf, training, validation
    )
    print("Fitting and predicting with matrix factorization...", flush=True)
    matrix_predictions, matrix_training, matrix_prediction = fit_and_predict(
        matrix_factorization, training, validation
    )
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    item_weight = optimal_blend_weight(
        actual, item_predictions, matrix_predictions
    )
    blended = blend_predictions(item_predictions, matrix_predictions, item_weight)
    grid_results = []
    for step in range(11):
        weight = step / 10
        grid_results.append(
            {
                "item_cf_weight": weight,
                "matrix_factorization_weight": 1.0 - weight,
                "rmse": rmse(
                    actual,
                    blend_predictions(item_predictions, matrix_predictions, weight),
                ),
            }
        )

    result = {
        "configuration": {
            "validation_ratio": 0.2,
            "seed": seed,
            "item_item_cf": ITEM_CF_CONFIGURATION,
            "matrix_factorization": matrix_factorization_configuration(seed),
        },
        "component_results": {
            "item_item_cf": {
                "rmse": rmse(actual, item_predictions),
                "training_seconds": item_training,
                "prediction_seconds": item_prediction,
            },
            "matrix_factorization": {
                "rmse": rmse(actual, matrix_predictions),
                "training_seconds": matrix_training,
                "prediction_seconds": matrix_prediction,
            },
        },
        "optimal_ensemble": {
            "item_cf_weight": item_weight,
            "matrix_factorization_weight": 1.0 - item_weight,
            "rmse": rmse(actual, blended),
        },
        "ensemble_resources": {
            "training_seconds": item_training + matrix_training,
            "prediction_seconds": item_prediction + matrix_prediction,
            "peak_memory_mb": peak_memory / (1024 * 1024),
        },
        "selected_final_weight": {
            "item_cf_weight": ENSEMBLE_ITEM_CF_WEIGHT,
            "matrix_factorization_weight": 1.0 - ENSEMBLE_ITEM_CF_WEIGHT,
        },
        "weight_grid": grid_results,
    }
    output = Path("output/experiments/ensemble_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
