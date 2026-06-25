from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_io import load_ratings


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    output = Path("fig")
    output.mkdir(parents=True, exist_ok=True)
    baseline = load_json(Path("output/experiments/baseline_results.json"))
    ensemble = load_json(Path("output/experiments/ensemble_results.json"))
    tuning = load_json(
        Path("output/experiments/matrix_factorization_tuning.json")
    )
    ratings = load_ratings(Path("data/train.txt"))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    blue = "#4C78A8"
    orange = "#F58518"
    green = "#54A24B"
    red = "#E45756"
    purple = "#7A5195"

    score_counts = Counter(int(score) for _, _, score in ratings)
    scores = sorted(score_counts)
    counts = [score_counts[score] for score in scores]
    plt.figure(figsize=(8.2, 4.5))
    bars = plt.bar([str(score) for score in scores], counts, color=blue)
    plt.xlabel("评分")
    plt.ylabel("评分记录数")
    plt.title("训练集评分分布")
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    save_figure(output / "rating_distribution.png")

    results = baseline["results"]
    names = ["全局均值", "偏置基线", "Item-CF", "矩阵分解", "融合模型"]
    rmse_values = [result["rmse"] for result in results]
    rmse_values.append(ensemble["optimal_ensemble"]["rmse"])
    colors = [blue, orange, green, red, purple]
    plt.figure(figsize=(8.2, 4.6))
    bars = plt.bar(names, rmse_values, color=colors)
    plt.ylabel("验证集 RMSE（越低越好）")
    plt.title("模型预测精度对比")
    plt.ylim(16.5, 21.3)
    plt.xticks(rotation=12)
    for bar, value in zip(bars, rmse_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.06,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_figure(output / "rmse_comparison.png")

    ensemble_resources = ensemble["ensemble_resources"]
    training_times = [result["training_seconds"] for result in results]
    training_times.append(ensemble_resources["training_seconds"])
    prediction_times = [result["prediction_seconds"] for result in results]
    prediction_times.append(ensemble_resources["prediction_seconds"])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].bar(names, training_times, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_title("训练时间")
    axes[0].set_ylabel("秒（对数刻度）")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(names, prediction_times, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_title("验证集预测时间")
    axes[1].set_ylabel("秒（对数刻度）")
    axes[1].tick_params(axis="x", rotation=25)
    figure.suptitle("启用内存跟踪时的运行时间对比")
    save_figure(output / "runtime_comparison.png")

    memory_values = [result["peak_memory_mb"] for result in results]
    memory_values.append(ensemble_resources["peak_memory_mb"])
    plt.figure(figsize=(8.2, 4.6))
    bars = plt.bar(names, memory_values, color=colors)
    plt.ylabel("Python 内存分配峰值（MB）")
    plt.title("模型峰值内存对比")
    plt.xticks(rotation=12)
    for bar, value in zip(bars, memory_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_figure(output / "memory_comparison.png")

    candidate_labels = [f"C{index}" for index in range(1, len(tuning) + 1)]
    tuning_rmse = [candidate["rmse"] for candidate in tuning]
    best_index = min(range(len(tuning)), key=lambda index: tuning_rmse[index])
    tuning_colors = [blue] * len(tuning)
    tuning_colors[best_index] = red
    plt.figure(figsize=(8.2, 4.4))
    bars = plt.bar(candidate_labels, tuning_rmse, color=tuning_colors)
    plt.ylabel("验证集 RMSE")
    plt.xlabel("候选参数组合")
    plt.title("矩阵分解超参数选择")
    plt.ylim(17.35, 17.49)
    for bar, value in zip(bars, tuning_rmse):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=45,
        )
    save_figure(output / "matrix_factorization_tuning.png")

    weight_grid = ensemble["weight_grid"]
    weights = [entry["item_cf_weight"] for entry in weight_grid]
    grid_rmse = [entry["rmse"] for entry in weight_grid]
    optimal = ensemble["optimal_ensemble"]
    plt.figure(figsize=(8.2, 4.5))
    plt.plot(weights, grid_rmse, marker="o", color=blue, label="步长为 0.1 的网格结果")
    plt.scatter(
        [optimal["item_cf_weight"]],
        [optimal["rmse"]],
        color=red,
        s=70,
        zorder=3,
        label=f"解析最优权重（{optimal['item_cf_weight']:.4f}）",
    )
    plt.xlabel("Item-CF 权重")
    plt.ylabel("验证集 RMSE")
    plt.title("融合模型权重选择")
    plt.legend()
    save_figure(output / "ensemble_weight_selection.png")


if __name__ == "__main__":
    main()
