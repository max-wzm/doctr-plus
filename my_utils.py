from time import sleep, time


def count_time(may_return_none=False):
    """
    return time cost as the last return value
    """

    def wrapper(func):
        def func_with_time(*args, **kwargs):
            t1 = time()
            ret = func(*args, **kwargs)
            t2 = time()
            time_cost = t2 - t1
            has_return = ret or may_return_none
            if not has_return:
                return time_cost
            if isinstance(ret, tuple):
                return *ret, time_cost
            return ret, time_cost

        return func_with_time

    return wrapper


@count_time()
def test(a):
    sleep(1)
    return 1, 2


t = test(0)
print(t)
