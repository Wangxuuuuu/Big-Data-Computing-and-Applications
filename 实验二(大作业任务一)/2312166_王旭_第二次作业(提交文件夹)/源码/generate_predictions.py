from __future__ import annotations

import argparse
import hashlib
import json
import time
from math import isfinite
from pathlib import Path

from src.data_io import load_queries, load_ratings, write_predictions
from src.item_cf import ItemItemCFModel
from src.matrix_factorization import BiasedMatrixFactorizationModel
from src.model_config import ENSEMBLE_ITEM_CF_WEIGHT, ITEM_CF_CONFIGURATION
from src.model_config import matrix_factorization_configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成正式测试集融合预测结果")
    parser.add_argument("--train", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--test", type=Path, default=Path("data/test.txt"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/final/result.txt")
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def validate_result(
    result_path: Path,
    expected_pairs: list[tuple[int, int]],
    minimum_score: float,
    maximum_score: float,
) -> dict[str, float | int | str]:
    result_ratings = load_ratings(result_path)
    result_pairs = [(user_id, item_id) for user_id, item_id, _ in result_ratings]
    if result_pairs != expected_pairs:
        raise ValueError("结果文件中的用户或物品顺序与 test.txt 不一致")

    scores = [score for _, _, score in result_ratings]
    if any(not isfinite(score) for score in scores):
        raise ValueError("结果文件包含非有限评分")
    if min(scores) < minimum_score or max(scores) > maximum_score:
        raise ValueError("结果文件包含超出训练评分范围的预测值")

    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    return {
        "predictions": len(scores),
        "minimum_prediction": min(scores),
        "maximum_prediction": max(scores),
        "mean_prediction": sum(scores) / len(scores),
        "sha256": digest,
    }


def main() -> None:
    args = parse_args()
    ratings = load_ratings(args.train)
    query_groups = load_queries(args.test)
    expected_pairs = [
        (group.user_id, item_id)
        for group in query_groups
        for item_id in group.item_ids
    ]
    scores = [score for _, _, score in ratings]
    minimum_score = min(scores)
    maximum_score = max(scores)

    item_cf = ItemItemCFModel(
        minimum_score,
        maximum_score,
        **ITEM_CF_CONFIGURATION,
    )
    matrix_configuration = matrix_factorization_configuration(args.seed)
    matrix_factorization = BiasedMatrixFactorizationModel(
        minimum_score,
        maximum_score,
        **matrix_configuration,
    )

    print("Fitting Item-CF on the complete training set...", flush=True)
    item_training_started = time.perf_counter()
    item_cf.fit(ratings)
    item_training_seconds = time.perf_counter() - item_training_started

    print("Fitting matrix factorization on the complete training set...", flush=True)
    matrix_training_started = time.perf_counter()
    matrix_factorization.fit(ratings)
    matrix_training_seconds = time.perf_counter() - matrix_training_started

    matrix_weight = 1.0 - ENSEMBLE_ITEM_CF_WEIGHT

    def predict(user_id: int, item_id: int) -> float:
        item_prediction = item_cf.predict(user_id, item_id)
        matrix_prediction = matrix_factorization.predict(user_id, item_id)
        return (
            ENSEMBLE_ITEM_CF_WEIGHT * item_prediction
            + matrix_weight * matrix_prediction
        )

    print("Predicting test pairs and writing the result file...", flush=True)
    prediction_started = time.perf_counter()
    write_predictions(args.output, query_groups, predict)
    prediction_seconds = time.perf_counter() - prediction_started
    validation = validate_result(
        args.output, expected_pairs, minimum_score, maximum_score
    )

    train_users = {user_id for user_id, _, _ in ratings}
    train_items = {item_id for _, item_id, _ in ratings}
    metadata = {
        "model": "item_item_cf_and_matrix_factorization_ensemble",
        "weights": {
            "item_item_cf": ENSEMBLE_ITEM_CF_WEIGHT,
            "matrix_factorization": matrix_weight,
        },
        "configuration": {
            "seed": args.seed,
            "item_item_cf": ITEM_CF_CONFIGURATION,
            "matrix_factorization": matrix_configuration,
        },
        "data": {
            "training_ratings": len(ratings),
            "test_users": len(query_groups),
            "test_pairs": len(expected_pairs),
            "new_test_users": len(
                {user_id for user_id, _ in expected_pairs} - train_users
            ),
            "new_test_items": len(
                {item_id for _, item_id in expected_pairs} - train_items
            ),
        },
        "timing_seconds": {
            "item_item_cf_training": item_training_seconds,
            "matrix_factorization_training": matrix_training_seconds,
            "prediction_and_writing": prediction_seconds,
        },
        "validation": validation,
    }
    metadata_path = args.output.with_name(f"{args.output.stem}_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
