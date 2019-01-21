import numba
import numpy as np

from timewindow.time_window_functions import get_agg_function


def time_window_df(df1, df2, date_left, date_right, groupby_left, groupby_right, value_right, window, agg_function_name):
    return time_window(
        df1[date_left].values,
        df1[groupby_left].values,
        df2[date_right].values,
        df2[groupby_right].values,
        df2[value_right].values,
        window,
        agg_function_name
    )


def time_window(date1: np.ndarray, groupby1: np.ndarray, date2: np.ndarray, groupby2: np.ndarray,
                value: np.ndarray, window: np.timedelta64, agg_function_name):

    order1 = date1.argsort()
    order2 = date2.argsort()

    date_all = np.concatenate((date1, date2))
    groupby_all = np.concatenate((groupby1, groupby2))
    value_all = np.concatenate((np.repeat(np.nan, len(date1)), value))

    order_all = date_all.argsort()

    agg_type, agg_function, agg_zero = get_agg_function(agg_function_name)

    result = tw(date_all[order_all], groupby_all[order_all], value_all[order_all], window, len(date1),
                date2[order2], groupby2[order2], value[order2], agg_type, agg_function, agg_zero)

    return result[order1.argsort()]


@numba.njit()
def tw(date_all: np.ndarray, groupby_all: np.ndarray, value_all: np.ndarray, window: np.timedelta64, len1: np.int64,
       date2: np.ndarray, groupby2: np.ndarray, value2: np.ndarray, agg_function, agg_function_param, agg_zero) -> np.ndarray:

    queue_start = 0
    queue_end = -1

    length = len(date_all)
    results = np.zeros(len1)
    results_end = -1

    for i in range(0, length):
        d = date_all[i]
        gb = groupby_all[i]
        v = value_all[i]

        while queue_start <= queue_end and (d - date2[queue_start]) > window:
            queue_start += 1

        if np.isnan(v):
            results_end += 1

            results[results_end] = agg_function(queue_start, queue_end, gb, groupby2, value2, agg_function_param, agg_zero)
        else:
            queue_end += 1

    return results
