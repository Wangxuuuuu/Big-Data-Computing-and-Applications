from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from typing import Protocol

from src.baselines import BiasBaselineModel, GlobalMeanModel
from src.data_io import Rating, load_queries, load_ratings, split_ratings_by_user
from src.data_io import summarize_dataset
from src.item_cf import ItemItemCFModel
from src.matrix_factorization import BiasedMatrixFactorizationModel
from src.metrics import rmse
from src.model_config import ITEM_CF_CONFIGURATION
from src.model_config import BIAS_BASELINE_CONFIGURATION
from src.model_config import matrix_factorization_configuration


class RatingModel(Protocol):
    def fit(self, ratings: list[Rating]) -> None: ...

    def predict(self, user_id: int, item_id: int) -> float: ...


def evaluate_model(
    name: str, model: RatingModel, training: list[Rating], validation: list[Rating]
) -> dict[str, float | str]:
    tracemalloc.start()
    training_started = time.perf_counter()
    model.fit(training)
    training_seconds = time.perf_counter() - training_started

    prediction_started = time.perf_counter()
    predictions = [model.predict(user_id, item_id) for user_id, item_id, _ in validation]
    prediction_seconds = time.perf_counter() - prediction_started
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    actual = [score for _, _, score in validation]
    return {
        "model": name,
        "rmse": rmse(actual, predictions),
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "peak_memory_mb": peak_memory / (1024 * 1024),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行推荐系统模型对比实验")
    parser.add_argument("--train", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--test", type=Path, default=Path("data/test.txt"))
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "global_mean",
            "bias_baseline",
            "item_item_cf",
            "matrix_factorization",
        ],
        default=[
            "global_mean",
            "bias_baseline",
            "item_item_cf",
            "matrix_factorization",
        ],
        help="选择要运行的模型；默认运行全部模型",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/experiments/baseline_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratings = load_ratings(args.train)
    query_groups = load_queries(args.test)
    training, validation = split_ratings_by_user(
        ratings, args.validation_ratio, args.seed
    )
    statistics = summarize_dataset(ratings)
    dataset_users = {user_id for user_id, _, _ in ratings}
    dataset_items = {item_id for _, item_id, _ in ratings}
    training_users = {user_id for user_id, _, _ in training}
    training_items = {item_id for _, item_id, _ in training}
    validation_pairs = [(user_id, item_id) for user_id, item_id, _ in validation]
    test_pairs = [
        (group.user_id, item_id)
        for group in query_groups
        for item_id in group.item_ids
    ]
    minimum_score = float(statistics["minimum_score"])
    maximum_score = float(statistics["maximum_score"])
    item_cf_configuration = ITEM_CF_CONFIGURATION.copy()
    matrix_factorization_config = matrix_factorization_configuration(args.seed)

    available_models: dict[str, RatingModel] = {
        "global_mean": GlobalMeanModel(minimum_score, maximum_score),
        "bias_baseline": BiasBaselineModel(
            minimum_score,
            maximum_score,
            **BIAS_BASELINE_CONFIGURATION,
        ),
        "item_item_cf": ItemItemCFModel(
            minimum_score,
            maximum_score,
            **item_cf_configuration,
        ),
        "matrix_factorization": BiasedMatrixFactorizationModel(
            minimum_score,
            maximum_score,
            **matrix_factorization_config,
        ),
    }
    models = [
        (model_name, available_models[model_name]) for model_name in args.models
    ]
    results = [
        evaluate_model(name, model, training, validation) for name, model in models
    ]

    payload = {
        "configuration": {
            "validation_ratio": args.validation_ratio,
            "seed": args.seed,
            "item_item_cf": item_cf_configuration,
            "matrix_factorization": matrix_factorization_config,
            "models": args.models,
        },
        "dataset": statistics,
        "split": {
            "training_ratings": len(training),
            "validation_ratings": len(validation),
            "test_users": len(query_groups),
            "test_pairs": sum(len(group.item_ids) for group in query_groups),
        },
        "cold_start": {
            "validation_pairs_with_new_user": sum(
                user_id not in training_users for user_id, _ in validation_pairs
            ),
            "validation_pairs_with_new_item": sum(
                item_id not in training_items for _, item_id in validation_pairs
            ),
            "test_users_not_in_full_training_data": len(
                {user_id for user_id, _ in test_pairs} - dataset_users
            ),
            "test_items_not_in_full_training_data": len(
                {item_id for _, item_id in test_pairs} - dataset_items
            ),
            "test_pairs_with_new_user": sum(
                user_id not in dataset_users for user_id, _ in test_pairs
            ),
            "test_pairs_with_new_item": sum(
                item_id not in dataset_items for _, item_id in test_pairs
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
