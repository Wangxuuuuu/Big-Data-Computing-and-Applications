from __future__ import annotations

from math import sqrt


def rmse(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("真实值与预测值数量不一致")
    if not actual:
        raise ValueError("RMSE 至少需要一条记录")

    squared_error = sum(
        (actual_score - predicted_score) ** 2
        for actual_score, predicted_score in zip(actual, predicted)
    )
    return sqrt(squared_error / len(actual))
