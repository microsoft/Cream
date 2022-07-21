import numpy as np
from numpy.random import Generator, PCG64

RNG = None


class AugRandomContext:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        global RNG
        assert RNG is None
        RNG = Generator(PCG64(seed=self.seed))

    def __exit__(self, *_):
        global RNG
        RNG = None


class random:
    # inline: random module
    @staticmethod
    def random():
        return RNG.random()

    @staticmethod
    def uniform(a, b):
        return random.random() * (b - a) + a

    @staticmethod
    def randint(a, b):
        # [low, high]
        return min(int(random.random() * (b - a + 1)) + a, b)

    @staticmethod
    def gauss(mu, sigma):
        return RNG.normal(mu, sigma)


class np_random:
    # numpy.random
    @staticmethod
    def choice(a, size, *args, **kwargs):
        return RNG.choice(a, size, *args, **kwargs)

    @staticmethod
    def randint(low, high, size=None, dtype=int):
        # [low, high)
        if size is None:
            return dtype(random.randint(low, high - 1))
        out = [random.randint(low, high - 1) for _ in range(size)]
        return np.array(out, dtype=dtype)

    @staticmethod
    def rand(*shape):
        return RNG.random(shape)

    @staticmethod
    def beta(a, b, size=None):
        return RNG.beta(a, b, size=size)


if __name__ == '__main__':
    for _ in range(2):
        with AugRandomContext(seed=0):
            print(np_random.randint(-100, 100, size=10))
        with AugRandomContext(seed=1):
            print(np_random.randint(-100, 100, size=10))
