import numpy as np


class OnePoint:
    def __init__(self, variables, probability=0.8):
        self._probability = probability
        self._value_bites = []
        current_bit = 0
        for key, value in variables.items():
            self._value_bites.append((current_bit, current_bit + value[0]))
            current_bit += value[0]

    def crossover(self, population):
        crossed_population = population.copy()
        for n in range(0, population.shape[0] // 2):
            # determine whether to do crossover
            if np.random.random() <= self._probability:
                for vb in self._value_bites:
                    # for every value find crossover point
                    point = np.random.randint(vb[0]+1, vb[1])
                    # for every value perform crossover
                    # join two parts of genes representing a value coming from two genotypes from selected population
                    p11 = population[2 * n, vb[0] : point]
                    p12 = population[2 * n + 1, point : vb[1]]
                    new1 = np.reshape(
                        np.concatenate(
                            (
                                p11,p12
                            )
                        ),
                        (1, -1),
                    )
                    p21 = population[2 * n + 1, vb[0] : point]
                    p22 = population[2 * n, point : vb[1]]
                    new2 = np.reshape(
                        np.concatenate(
                            (
                                p21, p22
                            )
                        ),
                        (1, -1),
                    )
                    crossed_population[2 * n, vb[0]: vb[1]] = new1
                    crossed_population[2 * n + 1, vb[0]: vb[1]] = new2
        return crossed_population


if __name__ == "__main__":
    pass
