import numpy as np
from random import sample


class Tournament:
    def __init__(self, n_tournaments=1):
        self._tournaments = n_tournaments

    def select(self, population, fitness):
        n = population.shape[0]
        selected_population = np.ndarray(shape=population.shape, dtype="I")
        for i in range(0, n):
            # generate n random distinct numbers
            candidates = sample(range(0, n), self._tournaments)
            # get candidate with the best fitness values
            best_fitness = fitness[candidates, :].min()
            # where returns array of 2d arrays with coordinates of matching values
            # first array is enough since only values matter in algorithm
            # and first dimension is needed as its a row
            winner_arg = np.where(fitness == best_fitness)[0][0]
            # assign winning genotype to the new population
            selected_population[i, :] = population[winner_arg, :]
        return selected_population


if __name__ == "__main__":
    s = Tournament(2)
    test = np.random.randint(2, size=(10, 5), dtype="I")
    test_fitness = np.random.random(size=(10, 1)) * 10 - 5
    # print(test, test_fitness)

    print(s.select(test, test_fitness))
