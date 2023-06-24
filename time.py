import self as self

import time
from gradient import batch_gd


class Time_score(self, x, y):
    pass


def time_score(x, y):
    start = time.time()
    batch_gd.fit(x, y)
    score = time.time() - start
    print('The time taken is', score)
