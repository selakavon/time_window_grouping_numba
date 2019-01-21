from io import StringIO
import numpy as np
import pandas as pd


class DataGenerator:
    def __init__(self, n, seed, window, window_unit):
        np.random.seed(seed)

        def col_f(n, r):
            return np.random.np.random.uniform(0, r, n)

        def col_n(n, r):
            return np.random.randint(0, r, n)

        def col_d(n):
            base = np.datetime64("2010-01-01")
            return np.array([base + np.timedelta64(np.random.randint(0, 3 * 365 * 24 * 3600), "s") for _ in range(0, n)])

        gb_max = n / 5
        v_max = 1000

        self.d1 = col_d(n)
        self.gb1 = col_n(n, gb_max)
        self.d2 = col_d(n)
        self.gb2 = col_n(n, gb_max)
        self.v = col_f(n, v_max)
        self.window = np.timedelta64(window, window_unit)

    def df1(self):
        return pd.DataFrame({
            "d1": self.d1,
            "gb1": self.gb1
        })

    def df2(self):
        return pd.DataFrame({
            "d2": self.d2,
            "gb2": self.gb2,
            "v": self.v
        })

    def __str__(self):
        s = StringIO()
        index = range(0, max(len(self.d1), len(self.d2)))
        for r in index:
            def p(c):
                v = c[r] if r < len(c) else ""
                s.write("{:>11} ".format(v))

            p(index)

            s.write(" | ")

            for c in [self.d1, self.gb1]:
                p(c)

            s.write(" | ")

            for c in [self.d2, self.gb2, self.v]:
                p(c)

            s.write("\n")

        return s.getvalue()