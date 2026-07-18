import unittest

import numpy as np
import pandas as pd

import app


class StrategyEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = app.make_demo_data("AAPL", 1, "5m")
        cls.data = app.add_indicators(raw).iloc[:2000].copy()

    def test_labels_are_matured_and_floor_is_unchanged(self):
        self.assertAlmostEqual(app.calculate_label_threshold(self.data.assign(return_1=0.0), 3), 0.0005)
        labels = app.make_labels(self.data, 3, threshold=0.0005)
        self.assertEqual(len(labels), len(self.data) - 3)
        self.assertEqual(labels.index[-1], self.data.index[-4])
        self.assertTrue(labels["target"].isin([0, 1]).all())

    def test_walk_forward_coverage_and_fresh_fold_boundaries(self):
        result = app.train_predict_model(self.data, "Random Forest", 1)
        self.assertFalse(result.fallback)
        self.assertEqual(result.prediction_start, app.ML_MIN_TRAIN_BARS + 1)
        self.assertEqual(result.probability.index.tolist(), self.data.index.tolist())
        self.assertTrue(result.probability.iloc[: app.ML_MIN_TRAIN_BARS + 1].isna().all())
        self.assertFalse(pd.isna(result.probability.iloc[-1]))
        self.assertGreaterEqual(result.walk_forward_folds, 2)
        self.assertGreater(result.prediction_coverage, 0.85)
        for _, fold in result.fold_metrics[result.fold_metrics["status"] == "trained"].iterrows():
            train_end = self.data.index.get_loc(fold["train_end"])
            predict_start = self.data.index.get_loc(fold["predict_start"])
            self.assertLess(train_end, predict_start)

    def test_early_probabilities_do_not_depend_on_future_tail(self):
        changed = self.data.copy()
        changed.loc[changed.index[-300:], "close"] *= 1.5
        result_a = app.train_predict_model(self.data, "Random Forest", 1)
        result_b = app.train_predict_model(changed, "Random Forest", 1)
        pd.testing.assert_series_equal(
            result_a.probability.iloc[app.ML_MIN_TRAIN_BARS + 1 : app.ML_MIN_TRAIN_BARS + 201].reset_index(drop=True),
            result_b.probability.iloc[app.ML_MIN_TRAIN_BARS + 1 : app.ML_MIN_TRAIN_BARS + 201].reset_index(drop=True),
            check_names=False,
        )

    def test_class_balance_ratio_allows_both_directions(self):
        self.assertEqual(app.class_balance_ratio(pd.Series([0] * 90 + [1] * 10)), 9.0)
        self.assertAlmostEqual(app.class_balance_ratio(pd.Series([0] * 10 + [1] * 90)), 1 / 9)
        self.assertIsNone(app.class_balance_ratio(pd.Series([0, 0, 0])))

    def test_backtest_shifts_signal_and_force_closes(self):
        index = pd.date_range("2026-01-01", periods=30, freq="5min")
        df = pd.DataFrame({"close": 100.0, "volume": 1.0}, index=index)
        signal = pd.Series("Hold", index=index, dtype=object)
        signal.iloc[0] = "Buy"
        limits = pd.Series(0.5, index=index)
        trades, _ = app.backtest(df, signal, 100_000, 0.2, limits, limits, 2.0, 80)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["entry_time"], index[1])
        self.assertEqual(trades.iloc[0]["exit_reason"], "end of window")


if __name__ == "__main__":
    unittest.main()
