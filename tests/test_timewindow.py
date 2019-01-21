import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, sampled_from
from scipy.stats import mode, skew

from tests.data_generator import DataGenerator
from timewindow.timewindow import time_window, time_window_df


@given(
    seed=integers(min_value=1, max_value=20),
    n_power=integers(1, 3),
    window=integers(1, 400),
    window_unit=sampled_from(["D", "s"]),
    agg_name = sampled_from(["sum", "min", "max", "median", "count", "nunique", "mean", "std", "mode", "skew"])
)
@settings(deadline=5000)
def test_time_window(seed, n_power, window, window_unit, agg_name):
    h = DataGenerator(10 ** n_power, seed, window, window_unit)
    results = time_window(h.d1, h.gb1, h.d2, h.gb2, h.v, h.window, agg_name)

    for i in range(0, len(results)):
        matched_values = h.v[(h.gb1[i] == h.gb2) & (h.d1[i] >= h.d2) & ((h.d1[i] - h.d2) <= h.window)]

        if agg_name == "sum":
            expected = matched_values.sum() if len(matched_values) > 0 else np.nan
        elif agg_name == "min":
            expected = matched_values.min() if len(matched_values) > 0 else np.nan
        elif agg_name == "max":
            expected = matched_values.max() if len(matched_values) > 0 else np.nan
        elif agg_name == "count":
            expected = len(matched_values)
        elif agg_name == "median":
            expected = np.median(matched_values) if len(matched_values) > 0 else np.nan
        elif agg_name == "nunique":
            expected = len(np.unique(matched_values))
        elif agg_name == "mean":
            expected = matched_values.mean() if len(matched_values) > 0 else np.nan
        elif agg_name == "std":
            expected = matched_values.std() if len(matched_values) > 0 else np.nan
        elif agg_name == "mode":
            expected = mode(matched_values.astype(np.int64)).mode[0] if len(matched_values) > 0 else np.nan
        elif agg_name == "skew":
            expected = skew(matched_values) if len(matched_values) > 0 else np.nan
        else:
            raise NotImplementedError

        assert np.isclose(expected, results[i], equal_nan=True), "\n" + "Failed for row: {}".format(i) + "\n" + str(h)


@given(
    agg_name = sampled_from(["sum", "min", "max", "median", "count", "nunique", "mean", "std", "mode", "skew"])
)
@settings(deadline=5000)
def test_time_window_dataframe(agg_name):
    h = DataGenerator(10 ** 3, 1, 5, "D")

    result_np = time_window(h.d1, h.gb1, h.d2, h.gb2, h.v, h.window, agg_name)
    result_df = time_window_df(df1=h.df1(), df2=h.df2(),
                               date_left="d1", date_right="d2", groupby_left="gb1", groupby_right="gb2", value_right="v",
                               window=h.window, agg_function_name=agg_name)

    assert np.allclose(result_df, result_np, equal_nan=True, atol=0, rtol=0)


@pytest.mark.parametrize("name", ["sum", "min", "max", "median", "count", "nunique", "mean", "std", "mode", "skew"])
def test_benchmark_time_window(benchmark, name):
    h = DataGenerator(10 ** 5, 1, 5, "D")
    benchmark(
        time_window,
        h.d1, h.gb1, h.d2, h.gb2, h.v, h.window, name
    )

