from __future__ import annotations

import json
import time
from pathlib import Path

from src.data_io import load_ratings, split_ratings_by_user
from src.matrix_factorization import BiasedMatrixFactorizationModel
from src.metrics import rmse


def main() -> None:
    ratings = load_ratings(Path("data/train.txt"))
    training, validation = split_ratings_by_user(ratings, 0.2, 2026)
    actual = [score for _, _, score in validation]
    candidates = [
        {
            "factors": 10,
            "epochs": 8,
            "learning_rate": 0.001,
            "factor_regularization": 0.1,
            "bias_regularization": 0.05,
        },
        {
            "factors": 10,
            "epochs": 12,
            "learning_rate": 0.001,
            "factor_regularization": 0.5,
            "bias_regularization": 0.1,
        },
        {
            "factors": 20,
            "epochs": 8,
            "learning_rate": 0.001,
            "factor_regularization": 0.5,
            "bias_regularization": 0.1,
        },
        {
            "factors": 10,
            "epochs": 12,
            "learning_rate": 0.0005,
            "factor_regularization": 0.1,
            "bias_regularization": 0.05,
        },
        {
            "factors": 5,
            "epochs": 8,
            "learning_rate": 0.002,
            "factor_regularization": 0.5,
            "bias_regularization": 0.1,
        },
        {
            "factors": 10,
            "epochs": 5,
            "learning_rate": 0.002,
            "factor_regularization": 0.5,
            "bias_regularization": 0.1,
        },
        {
            "factors": 10,
            "epochs": 8,
            "learning_rate": 0.002,
            "factor_regularization": 1.0,
            "bias_regularization": 0.1,
        },
        {
            "factors": 20,
            "epochs": 5,
            "learning_rate": 0.0015,
            "factor_regularization": 1.0,
            "bias_regularization": 0.1,
        },
    ]
    results = []

    for configuration in candidates:
        model = BiasedMatrixFactorizationModel(
            10.0,
            100.0,
            learning_rate_decay=0.95,
            seed=2026,
            **configuration,
        )
        started = time.perf_counter()
        model.fit(training)
        training_seconds = time.perf_counter() - started
        predictions = [
            model.predict(user_id, item_id)
            for user_id, item_id, _ in validation
        ]
        result = {
            **configuration,
            "rmse": rmse(actual, predictions),
            "training_seconds": training_seconds,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    output = Path("output/experiments/matrix_factorization_tuning.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
