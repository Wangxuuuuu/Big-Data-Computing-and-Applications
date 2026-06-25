import tempfile
import unittest
from pathlib import Path

from generate_predictions import validate_result
from src.baselines import BiasBaselineModel, GlobalMeanModel
from src.data_io import load_queries, load_ratings, split_ratings_by_user
from src.data_io import write_predictions
from src.ensemble import blend_predictions, optimal_blend_weight
from src.item_cf import ItemItemCFModel
from src.matrix_factorization import BiasedMatrixFactorizationModel
from src.metrics import rmse


class DataIoTests(unittest.TestCase):
    def test_load_ratings_and_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ratings_path = root / "ratings.txt"
            queries_path = root / "queries.txt"
            ratings_path.write_text("1|2\n10 20\n11 30\n2|1\n10 40\n", encoding="utf-8")
            queries_path.write_text("1|2\n12\n13\n", encoding="utf-8")

            self.assertEqual(
                load_ratings(ratings_path),
                [(1, 10, 20.0), (1, 11, 30.0), (2, 10, 40.0)],
            )
            groups = load_queries(queries_path)
            self.assertEqual(groups[0].user_id, 1)
            self.assertEqual(groups[0].item_ids, (12, 13))

            result_path = root / "result.txt"
            write_predictions(result_path, groups, lambda user_id, item_id: 75.0)
            self.assertEqual(
                result_path.read_text(encoding="utf-8"),
                "1|2\n12  75.000000\n13  75.000000\n",
            )
            result_summary = validate_result(
                result_path, [(1, 12), (1, 13)], 10.0, 100.0
            )
            self.assertEqual(result_summary["predictions"], 2)

    def test_split_is_reproducible_and_keeps_training_rating(self) -> None:
        ratings = [
            (1, 1, 10.0),
            (1, 2, 20.0),
            (1, 3, 30.0),
            (2, 1, 40.0),
            (2, 2, 50.0),
        ]
        first = split_ratings_by_user(ratings, 0.4, 7)
        second = split_ratings_by_user(ratings, 0.4, 7)
        self.assertEqual(first, second)
        self.assertEqual({user for user, _, _ in first[0]}, {1, 2})


class ModelTests(unittest.TestCase):
    def test_rmse_and_baselines(self) -> None:
        self.assertAlmostEqual(rmse([10.0, 20.0], [10.0, 30.0]), 50**0.5)
        ratings = [(1, 1, 20.0), (1, 2, 40.0), (2, 1, 60.0)]

        global_model = GlobalMeanModel(10.0, 100.0)
        global_model.fit(ratings)
        self.assertAlmostEqual(global_model.predict(9, 9), 40.0)

        bias_model = BiasBaselineModel(10.0, 100.0, epochs=2)
        bias_model.fit(ratings)
        prediction = bias_model.predict(1, 1)
        self.assertGreaterEqual(prediction, 10.0)
        self.assertLessEqual(prediction, 100.0)

    def test_item_item_cf_similarity_and_cold_start_fallback(self) -> None:
        ratings = [
            (1, 1, 10.0),
            (1, 2, 20.0),
            (2, 1, 20.0),
            (2, 2, 40.0),
            (3, 2, 30.0),
        ]
        model = ItemItemCFModel(
            10.0,
            100.0,
            neighbors=2,
            minimum_common_users=2,
            similarity_shrinkage=0.0,
        )
        model.fit(ratings)

        self.assertAlmostEqual(model.item_similarity(1, 2), 1.0)
        self.assertGreaterEqual(model.predict(3, 1), 10.0)
        self.assertLessEqual(model.predict(3, 1), 100.0)
        self.assertAlmostEqual(
            model.predict(99, 99), model.baseline.predict(99, 99)
        )

    def test_matrix_factorization_is_deterministic_and_handles_cold_start(self) -> None:
        ratings = [
            (1, 1, 20.0),
            (1, 2, 40.0),
            (2, 1, 60.0),
            (2, 2, 80.0),
        ]
        first = BiasedMatrixFactorizationModel(
            10.0, 100.0, factors=3, epochs=2, seed=7
        )
        second = BiasedMatrixFactorizationModel(
            10.0, 100.0, factors=3, epochs=2, seed=7
        )
        first.fit(ratings)
        second.fit(ratings)

        self.assertAlmostEqual(first.predict(1, 1), second.predict(1, 1))
        for user_id, item_id in [(1, 1), (99, 1), (1, 99), (99, 99)]:
            prediction = first.predict(user_id, item_id)
            self.assertGreaterEqual(prediction, 10.0)
            self.assertLessEqual(prediction, 100.0)

    def test_ensemble_weight_and_predictions(self) -> None:
        actual = [0.0, 10.0]
        first = [0.0, 0.0]
        second = [10.0, 10.0]
        weight = optimal_blend_weight(actual, first, second)

        self.assertAlmostEqual(weight, 0.5)
        self.assertEqual(blend_predictions(first, second, weight), [5.0, 5.0])


if __name__ == "__main__":
    unittest.main()
