from __future__ import annotations

from collections import defaultdict

from src.data_io import Rating


class GlobalMeanModel:
    def __init__(self, minimum_score: float, maximum_score: float) -> None:
        self.minimum_score = minimum_score
        self.maximum_score = maximum_score
        self.global_mean: float | None = None

    def fit(self, ratings: list[Rating]) -> None:
        if not ratings:
            raise ValueError("训练集不能为空")
        self.global_mean = sum(score for _, _, score in ratings) / len(ratings)

    def predict(self, user_id: int, item_id: int) -> float:
        del user_id, item_id
        if self.global_mean is None:
            raise RuntimeError("模型尚未训练")
        return self.global_mean


class BiasBaselineModel:
    def __init__(
        self,
        minimum_score: float,
        maximum_score: float,
        regularization: float = 10.0,
        epochs: int = 10,
    ) -> None:
        self.minimum_score = minimum_score
        self.maximum_score = maximum_score
        self.regularization = regularization
        self.epochs = epochs
        self.global_mean: float | None = None
        self.user_biases: dict[int, float] = {}
        self.item_biases: dict[int, float] = {}

    def fit(self, ratings: list[Rating]) -> None:
        if not ratings:
            raise ValueError("训练集不能为空")

        ratings_by_user: dict[int, list[tuple[int, float]]] = defaultdict(list)
        ratings_by_item: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for user_id, item_id, score in ratings:
            ratings_by_user[user_id].append((item_id, score))
            ratings_by_item[item_id].append((user_id, score))

        self.global_mean = sum(score for _, _, score in ratings) / len(ratings)
        self.user_biases = {user_id: 0.0 for user_id in ratings_by_user}
        self.item_biases = {item_id: 0.0 for item_id in ratings_by_item}

        for _ in range(self.epochs):
            for user_id, user_ratings in ratings_by_user.items():
                residual_sum = sum(
                    score - self.global_mean - self.item_biases[item_id]
                    for item_id, score in user_ratings
                )
                self.user_biases[user_id] = residual_sum / (
                    self.regularization + len(user_ratings)
                )

            for item_id, item_ratings in ratings_by_item.items():
                residual_sum = sum(
                    score - self.global_mean - self.user_biases[user_id]
                    for user_id, score in item_ratings
                )
                self.item_biases[item_id] = residual_sum / (
                    self.regularization + len(item_ratings)
                )

    def predict(self, user_id: int, item_id: int) -> float:
        if self.global_mean is None:
            raise RuntimeError("模型尚未训练")

        prediction = (
            self.global_mean
            + self.user_biases.get(user_id, 0.0)
            + self.item_biases.get(item_id, 0.0)
        )
        return min(self.maximum_score, max(self.minimum_score, prediction))
