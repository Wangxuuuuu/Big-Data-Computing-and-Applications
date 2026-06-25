from __future__ import annotations

from math import isfinite
from random import Random

from src.baselines import BiasBaselineModel
from src.data_io import Rating


class BiasedMatrixFactorizationModel:
    def __init__(
        self,
        minimum_score: float,
        maximum_score: float,
        factors: int = 20,
        epochs: int = 25,
        learning_rate: float = 0.005,
        learning_rate_decay: float = 0.95,
        factor_regularization: float = 0.05,
        bias_regularization: float = 0.02,
        seed: int = 2026,
    ) -> None:
        if factors < 1:
            raise ValueError("factors 必须大于 0")
        if epochs < 1:
            raise ValueError("epochs 必须大于 0")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate 必须大于 0")
        if not 0.0 < learning_rate_decay <= 1.0:
            raise ValueError("learning_rate_decay 必须在 0 和 1 之间")
        if factor_regularization < 0.0 or bias_regularization < 0.0:
            raise ValueError("正则化参数不能小于 0")

        self.minimum_score = minimum_score
        self.maximum_score = maximum_score
        self.factors = factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.factor_regularization = factor_regularization
        self.bias_regularization = bias_regularization
        self.seed = seed
        self.global_mean: float | None = None
        self.user_biases: dict[int, float] = {}
        self.item_biases: dict[int, float] = {}
        self.user_factors: dict[int, list[float]] = {}
        self.item_factors: dict[int, list[float]] = {}

    def fit(self, ratings: list[Rating]) -> None:
        if not ratings:
            raise ValueError("训练集不能为空")

        baseline = BiasBaselineModel(self.minimum_score, self.maximum_score)
        baseline.fit(ratings)
        if baseline.global_mean is None:
            raise RuntimeError("偏置基线训练失败")

        random = Random(self.seed)
        user_ids = sorted({user_id for user_id, _, _ in ratings})
        item_ids = sorted({item_id for _, item_id, _ in ratings})
        initialization_scale = 0.1 / self.factors**0.5

        self.global_mean = baseline.global_mean
        self.user_biases = baseline.user_biases.copy()
        self.item_biases = baseline.item_biases.copy()
        self.user_factors = {
            user_id: [
                random.uniform(-initialization_scale, initialization_scale)
                for _ in range(self.factors)
            ]
            for user_id in user_ids
        }
        self.item_factors = {
            item_id: [
                random.uniform(-initialization_scale, initialization_scale)
                for _ in range(self.factors)
            ]
            for item_id in item_ids
        }

        shuffled_ratings = ratings.copy()
        current_learning_rate = self.learning_rate
        for _ in range(self.epochs):
            random.shuffle(shuffled_ratings)
            for user_id, item_id, score in shuffled_ratings:
                user_vector = self.user_factors[user_id]
                item_vector = self.item_factors[item_id]
                interaction = sum(
                    user_value * item_value
                    for user_value, item_value in zip(user_vector, item_vector)
                )
                prediction = (
                    self.global_mean
                    + self.user_biases[user_id]
                    + self.item_biases[item_id]
                    + interaction
                )
                error = score - prediction

                self.user_biases[user_id] += current_learning_rate * (
                    error - self.bias_regularization * self.user_biases[user_id]
                )
                self.item_biases[item_id] += current_learning_rate * (
                    error - self.bias_regularization * self.item_biases[item_id]
                )

                for factor_index in range(self.factors):
                    user_value = user_vector[factor_index]
                    item_value = item_vector[factor_index]
                    user_vector[factor_index] += current_learning_rate * (
                        error * item_value
                        - self.factor_regularization * user_value
                    )
                    item_vector[factor_index] += current_learning_rate * (
                        error * user_value
                        - self.factor_regularization * item_value
                    )

                if not isfinite(error):
                    raise RuntimeError("矩阵分解训练发散，请降低学习率")
            current_learning_rate *= self.learning_rate_decay

    def predict(self, user_id: int, item_id: int) -> float:
        if self.global_mean is None:
            raise RuntimeError("模型尚未训练")

        prediction = (
            self.global_mean
            + self.user_biases.get(user_id, 0.0)
            + self.item_biases.get(item_id, 0.0)
        )
        user_vector = self.user_factors.get(user_id)
        item_vector = self.item_factors.get(item_id)
        if user_vector is not None and item_vector is not None:
            prediction += sum(
                user_value * item_value
                for user_value, item_value in zip(user_vector, item_vector)
            )

        return min(self.maximum_score, max(self.minimum_score, prediction))
