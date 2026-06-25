from __future__ import annotations


def blend_predictions(
    first: list[float], second: list[float], first_weight: float
) -> list[float]:
    if len(first) != len(second):
        raise ValueError("两组预测值数量不一致")
    if not 0.0 <= first_weight <= 1.0:
        raise ValueError("融合权重必须在 0 和 1 之间")

    second_weight = 1.0 - first_weight
    return [
        first_weight * first_value + second_weight * second_value
        for first_value, second_value in zip(first, second)
    ]


def optimal_blend_weight(
    actual: list[float], first: list[float], second: list[float]
) -> float:
    if len(actual) != len(first) or len(actual) != len(second):
        raise ValueError("真实值与两组预测值数量不一致")
    if not actual:
        raise ValueError("融合至少需要一条记录")

    differences = [
        first_value - second_value
        for first_value, second_value in zip(first, second)
    ]
    denominator = sum(difference**2 for difference in differences)
    if denominator == 0.0:
        return 0.5

    numerator = sum(
        (actual_value - second_value) * difference
        for actual_value, second_value, difference in zip(
            actual, second, differences
        )
    )
    return min(1.0, max(0.0, numerator / denominator))
