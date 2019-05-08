import numpy as np


class Uniform:
    def __init__(self, size=(0, 0)):
        self._population_size = size[0]
        self._genotype_length = size[1]

    # randomly fills matrix with 0s and 1s
    def initialize_population(self):
        return np.random.randint(
            2, size=(self._population_size, self._genotype_length), dtype="I"
        )


if __name__ == "__main__":
    m = 10000
    n = 10000
    uni = Uniform((m, n))

    assert (uni.initialize_population() <= 1).all() is True
    assert (uni.initialize_population() >= 0).all() is True
    assert uni.initialize_population().shape == (m, n)
