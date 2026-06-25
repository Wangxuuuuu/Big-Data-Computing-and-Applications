from __future__ import annotations

import json
from pathlib import Path

from src.model_config import BIAS_BASELINE_CONFIGURATION


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def reduction_percent(reference: float, result: float) -> float:
    return (reference - result) / reference * 100.0


def main() -> None:
    experiment_dir = Path("output/experiments")
    baseline = load_json(experiment_dir / "baseline_results.json")
    ensemble = load_json(experiment_dir / "ensemble_results.json")
    tuning = load_json(experiment_dir / "matrix_factorization_tuning.json")
    final_result = load_json(Path("output/final/result_metadata.json"))

    single_results = {result["model"]: result for result in baseline["results"]}
    ensemble_result = {
        "model": "ensemble",
        "rmse": ensemble["optimal_ensemble"]["rmse"],
        **ensemble["ensemble_resources"],
    }
    model_results = [
        single_results["global_mean"],
        single_results["bias_baseline"],
        single_results["item_item_cf"],
        single_results["matrix_factorization"],
        ensemble_result,
    ]
    best_tuning = min(tuning, key=lambda result: result["rmse"])

    global_rmse = single_results["global_mean"]["rmse"]
    bias_rmse = single_results["bias_baseline"]["rmse"]
    item_rmse = single_results["item_item_cf"]["rmse"]
    matrix_rmse = single_results["matrix_factorization"]["rmse"]
    ensemble_rmse = ensemble_result["rmse"]

    summary = {
        "evaluation_protocol": {
            "split": "per-user holdout",
            "validation_ratio": baseline["configuration"]["validation_ratio"],
            "seed": baseline["configuration"]["seed"],
            "metric": "RMSE",
            "runtime_environment": "Windows, Python 3.8.0",
            "resource_measurement": "Python tracemalloc peak allocated memory",
        },
        "dataset": baseline["dataset"],
        "split": baseline["split"],
        "cold_start": baseline["cold_start"],
        "model_configuration": {
            "global_mean": {},
            "bias_baseline": BIAS_BASELINE_CONFIGURATION,
            "item_item_cf": baseline["configuration"]["item_item_cf"],
            "matrix_factorization": baseline["configuration"][
                "matrix_factorization"
            ],
            "ensemble": {
                "item_item_cf_weight": ensemble["optimal_ensemble"][
                    "item_cf_weight"
                ],
                "matrix_factorization_weight": ensemble["optimal_ensemble"][
                    "matrix_factorization_weight"
                ],
            },
        },
        "model_results": model_results,
        "matrix_factorization_tuning": {
            "candidate_count": len(tuning),
            "selected": best_tuning,
        },
        "relative_rmse_reduction_percent": {
            "bias_baseline_vs_global_mean": reduction_percent(
                global_rmse, bias_rmse
            ),
            "item_item_cf_vs_bias_baseline": reduction_percent(
                bias_rmse, item_rmse
            ),
            "matrix_factorization_vs_bias_baseline": reduction_percent(
                bias_rmse, matrix_rmse
            ),
            "ensemble_vs_item_item_cf": reduction_percent(
                item_rmse, ensemble_rmse
            ),
            "ensemble_vs_global_mean": reduction_percent(
                global_rmse, ensemble_rmse
            ),
        },
        "final_prediction": final_result,
    }

    output_dir = Path("output/summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "experiment_data_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dataset = summary["dataset"]
    split = summary["split"]
    cold_start = summary["cold_start"]
    reductions = summary["relative_rmse_reduction_percent"]
    model_names = {
        "global_mean": "全局均值",
        "bias_baseline": "偏置基线",
        "item_item_cf": "Item-CF",
        "matrix_factorization": "矩阵分解",
        "ensemble": "融合模型",
    }

    lines = [
        "# 实验数据汇总",
        "",
        "## 1. 评价设置",
        "",
        "- 划分方法：按用户留出 20% 评分作为验证集。",
        f"- 随机种子：{baseline['configuration']['seed']}。",
        "- 评价指标：RMSE，数值越低越好。",
        "- 运行环境：Windows、Python 3.8.0。",
        "- 时间与内存：均在启用 `tracemalloc` 的实验中测得，测量口径一致；运行时间会受机器负载影响。",
        "",
        "## 2. 数据规模与稀疏度",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 用户数 | {dataset['users']} |",
        f"| 物品数 | {dataset['items']} |",
        f"| 评分数 | {dataset['ratings']} |",
        f"| 最低/最高评分 | {dataset['minimum_score']:.0f} / {dataset['maximum_score']:.0f} |",
        f"| 平均评分 | {dataset['mean_score']:.4f} |",
        f"| 密度 | {dataset['density'] * 100:.4f}% |",
        f"| 稀疏度 | {dataset['sparsity'] * 100:.4f}% |",
        f"| 训练子集评分数 | {split['training_ratings']} |",
        f"| 验证集评分数 | {split['validation_ratings']} |",
        f"| 正式测试用户数 | {split['test_users']} |",
        f"| 正式测试预测对数 | {split['test_pairs']} |",
        "",
        "## 3. 冷启动情况",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 验证集新用户评分对 | {cold_start['validation_pairs_with_new_user']} |",
        f"| 验证集新物品评分对 | {cold_start['validation_pairs_with_new_item']} |",
        f"| 正式测试新用户数 | {cold_start['test_users_not_in_full_training_data']} |",
        f"| 正式测试新物品数 | {cold_start['test_items_not_in_full_training_data']} |",
        f"| 涉及新用户的测试对 | {cold_start['test_pairs_with_new_user']} |",
        f"| 涉及新物品的测试对 | {cold_start['test_pairs_with_new_item']} |",
        "",
        f"验证集中约 {cold_start['validation_pairs_with_new_item'] / split['validation_ratings'] * 100:.2f}% 的评分对涉及训练子集中未出现的物品；正式测试集中约 {cold_start['test_pairs_with_new_user'] / split['test_pairs'] * 100:.2f}% 的预测对涉及新用户，约 {cold_start['test_pairs_with_new_item'] / split['test_pairs'] * 100:.2f}% 涉及新物品。",
        "",
        "涉及新用户与新物品的测试对可能重叠，因此不能直接将两项相加。所有模型均使用全局均值或偏置项进行冷启动兜底。",
        "",
        "## 4. 模型参数",
        "",
        f"- 偏置基线：正则化系数 {BIAS_BASELINE_CONFIGURATION['regularization']}，迭代 {BIAS_BASELINE_CONFIGURATION['epochs']} 轮。",
        "- Item-CF：邻居数 20，共同评分用户至少 2 人，相似度收缩系数 10，缓存上限 250000。",
        "- 矩阵分解：5 个潜因子，8 轮，初始学习率 0.002，衰减率 0.95，潜因子正则化 0.5，偏置正则化 0.1。",
        f"- 融合模型：Item-CF 权重 {ensemble['optimal_ensemble']['item_cf_weight']:.4f}，矩阵分解权重 {ensemble['optimal_ensemble']['matrix_factorization_weight']:.4f}。",
        "",
        "矩阵分解共比较 8 组参数，最终选择验证集 RMSE 最低的配置。融合权重通过最小化验证集平方误差解析求得。",
        "",
        "## 5. 模型实验结果",
        "",
        "| 模型 | RMSE | 训练时间/s | 预测时间/s | 峰值内存/MB |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in model_results:
        lines.append(
            f"| {model_names[result['model']]} | {result['rmse']:.4f} | "
            f"{result['training_seconds']:.4f} | {result['prediction_seconds']:.4f} | "
            f"{result['peak_memory_mb']:.2f} |"
        )

    lines.extend(
        [
            "",
            "峰值内存为 `tracemalloc` 记录的 Python 内存分配峰值，不等同于操作系统显示的进程总内存。融合模型的训练与预测时间包含两个组成模型。",
            "",
            "## 6. 结果结论",
            "",
            f"- 偏置基线相对全局均值降低 RMSE {reductions['bias_baseline_vs_global_mean']:.2f}%。",
            f"- Item-CF 相对偏置基线降低 RMSE {reductions['item_item_cf_vs_bias_baseline']:.2f}%。",
            f"- 矩阵分解相对偏置基线降低 RMSE {reductions['matrix_factorization_vs_bias_baseline']:.2f}%。",
            f"- 融合模型相对 Item-CF 降低 RMSE {reductions['ensemble_vs_item_item_cf']:.2f}%。",
            f"- 融合模型相对全局均值累计降低 RMSE {reductions['ensemble_vs_global_mean']:.2f}%。",
            "- Item-CF 是最准确的单模型，但预测时间与内存最高；矩阵分解预测快、内存低；融合模型取得最低 RMSE，但总体资源消耗由 Item-CF 主导。",
            "",
            "## 7. 正式测试预测",
            "",
            f"- 使用完整 {final_result['data']['training_ratings']} 条评分重新训练。",
            f"- 输出 {final_result['validation']['predictions']} 条预测。",
            f"- 预测范围：{final_result['validation']['minimum_prediction']:.6f}～{final_result['validation']['maximum_prediction']:.6f}。",
            f"- 平均预测：{final_result['validation']['mean_prediction']:.6f}。",
            f"- 结果文件 SHA-256：`{final_result['validation']['sha256']}`。",
            "- 正式测试集没有真实评分，因此无法在本地计算正式测试 RMSE。",
            "",
        ]
    )
    markdown_path = output_dir / "experiment_data_summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(markdown_path)
    print(json_path)


if __name__ == "__main__":
    main()
