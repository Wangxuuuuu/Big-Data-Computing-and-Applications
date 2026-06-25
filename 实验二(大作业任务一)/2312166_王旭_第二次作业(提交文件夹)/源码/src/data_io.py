from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable, Tuple


Rating = Tuple[int, int, float]


@dataclass(frozen=True)
class QueryGroup:
    user_id: int
    item_ids: tuple[int, ...]


def load_ratings(path: Path) -> list[Rating]:
    ratings: list[Rating] = []
    current_user: int | None = None
    expected_count = 0
    actual_count = 0

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        if "|" in line:
            if current_user is not None and actual_count != expected_count:
                raise ValueError(
                    f"{path}:{line_number}: 用户 {current_user} 声明 {expected_count} 条评分，"
                    f"实际读取 {actual_count} 条"
                )
            user_text, count_text = line.split("|", maxsplit=1)
            current_user = int(user_text)
            expected_count = int(count_text)
            actual_count = 0
            continue

        if current_user is None:
            raise ValueError(f"{path}:{line_number}: 评分记录之前缺少用户头")

        item_text, score_text = line.split()
        ratings.append((current_user, int(item_text), float(score_text)))
        actual_count += 1

    if current_user is not None and actual_count != expected_count:
        raise ValueError(
            f"{path}: 用户 {current_user} 声明 {expected_count} 条评分，"
            f"实际读取 {actual_count} 条"
        )

    if not ratings:
        raise ValueError(f"{path}: 未读取到评分")
    return ratings


def load_queries(path: Path) -> list[QueryGroup]:
    groups: list[QueryGroup] = []
    current_user: int | None = None
    expected_count = 0
    item_ids: list[int] = []

    def append_current_group() -> None:
        if current_user is None:
            return
        if len(item_ids) != expected_count:
            raise ValueError(
                f"{path}: 用户 {current_user} 声明 {expected_count} 个物品，"
                f"实际读取 {len(item_ids)} 个"
            )
        groups.append(QueryGroup(current_user, tuple(item_ids)))

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        if "|" in line:
            append_current_group()
            user_text, count_text = line.split("|", maxsplit=1)
            current_user = int(user_text)
            expected_count = int(count_text)
            item_ids = []
            continue

        if current_user is None:
            raise ValueError(f"{path}:{line_number}: 物品记录之前缺少用户头")
        item_ids.append(int(line))

    append_current_group()
    if not groups:
        raise ValueError(f"{path}: 未读取到待预测记录")
    return groups


def split_ratings_by_user(
    ratings: list[Rating], validation_ratio: float, seed: int
) -> tuple[list[Rating], list[Rating]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio 必须在 0 和 1 之间")

    ratings_by_user: dict[int, list[Rating]] = defaultdict(list)
    for rating in ratings:
        ratings_by_user[rating[0]].append(rating)

    random = Random(seed)
    training: list[Rating] = []
    validation: list[Rating] = []

    for user_id in sorted(ratings_by_user):
        user_ratings = ratings_by_user[user_id]
        if len(user_ratings) < 2:
            training.extend(user_ratings)
            continue

        validation_count = max(1, round(len(user_ratings) * validation_ratio))
        validation_count = min(validation_count, len(user_ratings) - 1)
        validation_indices = set(
            random.sample(range(len(user_ratings)), validation_count)
        )

        for index, rating in enumerate(user_ratings):
            target = validation if index in validation_indices else training
            target.append(rating)

    return training, validation


def summarize_dataset(ratings: list[Rating]) -> dict[str, float | int]:
    users = {user_id for user_id, _, _ in ratings}
    items = {item_id for _, item_id, _ in ratings}
    scores = [score for _, _, score in ratings]
    possible_ratings = len(users) * len(items)
    density = len(ratings) / possible_ratings if possible_ratings else 0.0

    return {
        "users": len(users),
        "items": len(items),
        "ratings": len(ratings),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "mean_score": sum(scores) / len(scores),
        "density": density,
        "sparsity": 1.0 - density,
    }


def write_predictions(
    path: Path,
    groups: list[QueryGroup],
    predict: Callable[[int, int], float],
) -> None:
    lines: list[str] = []
    for group in groups:
        lines.append(f"{group.user_id}|{len(group.item_ids)}")
        for item_id in group.item_ids:
            lines.append(f"{item_id}  {predict(group.user_id, item_id):.6f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
