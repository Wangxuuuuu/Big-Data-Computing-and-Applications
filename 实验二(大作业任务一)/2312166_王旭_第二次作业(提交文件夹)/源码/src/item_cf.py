from __future__ import annotations

from collections import OrderedDict, defaultdict
from heapq import nlargest
from math import sqrt

from src.baselines import BiasBaselineModel
from src.data_io import Rating


class ItemItemCFModel:
    def __init__(
        self,
        minimum_score: float,
        maximum_score: float,
        neighbors: int = 20,
        minimum_common_users: int = 2,
        similarity_shrinkage: float = 10.0,
        similarity_cache_size: int = 250_000,
    ) -> None:
        if neighbors < 1:
            raise ValueError("neighbors 必须大于 0")
        if minimum_common_users < 1:
            raise ValueError("minimum_common_users 必须大于 0")
        if similarity_shrinkage < 0.0:
            raise ValueError("similarity_shrinkage 不能小于 0")
        if similarity_cache_size < 1:
            raise ValueError("similarity_cache_size 必须大于 0")

        self.minimum_score = minimum_score
        self.maximum_score = maximum_score
        self.neighbors = neighbors
        self.minimum_common_users = minimum_common_users
        self.similarity_shrinkage = similarity_shrinkage
        self.similarity_cache_size = similarity_cache_size
        self.baseline = BiasBaselineModel(minimum_score, maximum_score)
        self.user_ratings: dict[int, dict[int, float]] = {}
        self.item_ratings: dict[int, dict[int, float]] = {}
        self.item_means: dict[int, float] = {}
        self.similarity_cache: OrderedDict[tuple[int, int], float] = OrderedDict()

    def fit(self, ratings: list[Rating]) -> None:
        if not ratings:
            raise ValueError("训练集不能为空")

        user_ratings: dict[int, dict[int, float]] = defaultdict(dict)
        item_ratings: dict[int, dict[int, float]] = defaultdict(dict)
        for user_id, item_id, score in ratings:
            user_ratings[user_id][item_id] = score
            item_ratings[item_id][user_id] = score

        self.baseline.fit(ratings)
        self.user_ratings = dict(user_ratings)
        self.item_ratings = dict(item_ratings)
        self.item_means = {
            item_id: sum(scores.values()) / len(scores)
            for item_id, scores in self.item_ratings.items()
        }
        self.similarity_cache.clear()

    def item_similarity(self, first_item: int, second_item: int) -> float:
        if first_item == second_item:
            return 1.0

        key = (
            (first_item, second_item)
            if first_item < second_item
            else (second_item, first_item)
        )
        cached = self.similarity_cache.get(key)
        if cached is not None:
            self.similarity_cache.move_to_end(key)
            return cached

        first_ratings = self.item_ratings.get(first_item)
        second_ratings = self.item_ratings.get(second_item)
        if first_ratings is None or second_ratings is None:
            return 0.0

        if len(first_ratings) > len(second_ratings):
            first_item, second_item = second_item, first_item
            first_ratings, second_ratings = second_ratings, first_ratings

        first_mean = self.item_means[first_item]
        second_mean = self.item_means[second_item]
        numerator = 0.0
        first_squared = 0.0
        second_squared = 0.0
        common_users = 0

        for user_id, first_score in first_ratings.items():
            second_score = second_ratings.get(user_id)
            if second_score is None:
                continue
            first_deviation = first_score - first_mean
            second_deviation = second_score - second_mean
            numerator += first_deviation * second_deviation
            first_squared += first_deviation**2
            second_squared += second_deviation**2
            common_users += 1

        denominator = sqrt(first_squared * second_squared)
        if common_users < self.minimum_common_users or denominator == 0.0:
            similarity = 0.0
        else:
            shrinkage = common_users / (
                common_users + self.similarity_shrinkage
            )
            similarity = numerator / denominator * shrinkage

        self.similarity_cache[key] = similarity
        if len(self.similarity_cache) > self.similarity_cache_size:
            self.similarity_cache.popitem(last=False)
        return similarity

    def predict(self, user_id: int, item_id: int) -> float:
        baseline_prediction = self.baseline.predict(user_id, item_id)
        rated_items = self.user_ratings.get(user_id)
        if rated_items is None or item_id not in self.item_ratings:
            return baseline_prediction

        candidates: list[tuple[float, float, int, float]] = []
        for neighbor_item, score in rated_items.items():
            if neighbor_item == item_id:
                continue
            similarity = self.item_similarity(item_id, neighbor_item)
            if similarity != 0.0:
                candidates.append(
                    (abs(similarity), similarity, neighbor_item, score)
                )

        nearest_neighbors = nlargest(self.neighbors, candidates)
        if not nearest_neighbors:
            return baseline_prediction

        weighted_residual = 0.0
        similarity_sum = 0.0
        for _, similarity, neighbor_item, score in nearest_neighbors:
            neighbor_baseline = self.baseline.predict(user_id, neighbor_item)
            weighted_residual += similarity * (score - neighbor_baseline)
            similarity_sum += abs(similarity)

        prediction = baseline_prediction + weighted_residual / similarity_sum
        return min(self.maximum_score, max(self.minimum_score, prediction))
