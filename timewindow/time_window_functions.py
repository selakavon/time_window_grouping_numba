import numba
import numpy as np


def get_agg_function(name: str):
    return {
        "sum": (agg_seq, f_sum, np.nan),
        "max": (agg_seq, f_max, np.nan),
        "min": (agg_seq, f_min, np.nan),
        "count": (agg_count, np.nan, 0),
        "median": (agg_median, np.nan, np.nan),
        "nunique": (agg_nunique, np.nan, 0),
        "mean": (agg_mean, np.nan, np.nan),
        "std": (agg_std, np.nan, np.nan),
        "mode": (agg_mode, np.nan, np.nan),
        "skew": (agg_skew, np.nan, np.nan)
    }.get(name)


@numba.njit()
def skew(a):
    if len(a) < 3:
        return 0

    a_zero_mean = a - np.mean(a)

    s2 = a_zero_mean ** 2
    s3 = s2 * a_zero_mean

    m2 = s2.mean()
    m3 = s3.mean()

    return m3 / m2 ** 1.5


@numba.njit()
def agg_skew(queue_start, queue_end, gb, groupby2, value2, combine, agg_zero):
    queue_idx = np.arange(queue_start, queue_end + 1)
    queue_groupby2 = groupby2[queue_idx]
    queue_value2 = value2[queue_idx]

    same_group_idx = gb == queue_groupby2

    values = queue_value2[same_group_idx]

    if len(values) > 0:
        return skew(values)
    else:
        return agg_zero


@numba.njit()
def agg_mode(queue_start, queue_end, gb, groupby2, value2, combine, agg_zero):
    queue_idx = np.arange(queue_start, queue_end + 1)
    queue_groupby2 = groupby2[queue_idx]
    queue_value2 = value2[queue_idx]

    same_group_idx = gb == queue_groupby2

    values = queue_value2[same_group_idx]

    if len(values) > 0:
        int_values = values.astype(np.int64)
        bincount = np.bincount(int_values)
        bincount = bincount[bincount != 0]
        return np.unique(int_values)[bincount.argmax()]
    else:
        return agg_zero


@numba.njit()
def agg_nunique(queue_start, queue_end, gb, groupby2, value2, combine, agg_zero):
    queue_idx = np.arange(queue_start, queue_end + 1)
    queue_groupby2 = groupby2[queue_idx]
    queue_value2 = value2[queue_idx]

    same_group_idx = gb == queue_groupby2

    values = queue_value2[same_group_idx]

    if len(values) > 0:
        return len(np.unique(values))
    else:
        return agg_zero


@numba.njit()
def agg_median(queue_start, queue_end, gb, groupby2, value2, combine, agg_zero):
    queue_idx = np.arange(queue_start, queue_end + 1)
    queue_groupby2 = groupby2[queue_idx]
    queue_value2 = value2[queue_idx]

    same_group_idx = gb == queue_groupby2

    values = queue_value2[same_group_idx]

    if len(values) > 0:
        return np.median(values)
    else:
        return agg_zero


@numba.njit()
def agg_std(queue_start, queue_end, gb, groupby2, value2, combine, agg_zero):
    queue_idx = np.arange(queue_start, queue_end + 1)
    queue_groupby2 = groupby2[queue_idx]
    queue_value2 = value2[queue_idx]

    same_group_idx = gb == queue_groupby2

    values = queue_value2[same_group_idx]

    if len(values) > 0:
        return values.std()
    else:
        return agg_zero


@numba.njit()
def agg_mean(queue_start, queue_end, gb, groupby2, value2, combine, agg_zero):
    count = 0
    sum = 0
    empty = True
    for j in range(queue_start, queue_end + 1):
        if gb == groupby2[j]:
            count += 1
            sum += value2[j]
            empty = False

    if empty:
        return agg_zero
    else:
        return sum / count


@numba.njit()
def agg_count(queue_start, queue_end, gb, groupby2, value2, combine, agg_zero):
    count = 0
    for j in range(queue_start, queue_end + 1):
        if gb == groupby2[j]:
            count += 1

    return count


@numba.njit()
def agg_seq(queue_start, queue_end, gb, groupby2, value2, combine, agg_none):
    result = agg_none
    first = True
    for j in range(queue_start, queue_end + 1):
        if gb == groupby2[j]:
            if first:
                result = value2[j]
                first = False
            else:
                result = combine(result, value2[j])

    return result


@numba.njit()
def f_sum(a, b):
    return a + b


@numba.njit()
def f_max(a, b):
    return a if a > b else b


@numba.njit()
def f_min(a, b):
    return a if a < b else b