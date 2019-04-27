import numpy as np


class UniformMutation:
    def __init__(self, probability=0.01):
        self._probability = probability

    def mutate(self, population):
        mask = np.random.choice(
            (True, False),
            size=population.shape,
            p=[self._probability, 1 - self._probability],
        )
        mutated_population = population.copy()
        mutated_population[mask] = 1 - mutated_population[mask]
        return mutated_population


if __name__ == "__main__":
    test = np.random.randint(2, size=(100, 100), dtype="I")

    um = UniformMutation(0.1)
    m = um.mutate(test)

    assert not np.array_equal(m, test)
