import numpy as np
from collections import OrderedDict


def chromosomes_to_value(chromosomes, first_bit, last_bit, min_value, max_value):
    value_representation = chromosomes[:, first_bit:last_bit]
    L = value_representation.shape[1]
    bit_weights = np.zeros(shape=(1, L), dtype="I")
    for i in range(0, L):
        bit_weights[0, i] = pow(2, L - i - 1)
    float_vals = np.multiply(value_representation.astype("float"), bit_weights)
    float_vals = np.sum(float_vals, axis=1, keepdims=True)
    float_vals = min_value + ((max_value - min_value) / (pow(2, L) - 1)) * float_vals
    return float_vals


def x_square(v):
    return v["x"] * v["x"]


def himmelblau(v):
    return (v["x"] ** 2 + v["y"] - 11) ** 2 + (v["x"] + v["y"] ** 2 - 7) ** 2


class FitnessFunction:
    def __init__(self, variables, expression, linear_scaling=False):
        self._variables = OrderedDict(variables)
        self._fitness_function = expression
        self._scale = linear_scaling

    def chromosomes_to_values(self, population):
        values = OrderedDict()
        current_bit = 0
        for key, value in self._variables.items():
            values[key] = chromosomes_to_value(
                population, current_bit, current_bit + value[0], value[1], value[2]
            )
            current_bit += value[0]
        return values

    def calculate_fitness(self, population):
        variables = self.chromosomes_to_values(population)
        fitness = self._fitness_function(variables)
        if self._scale:
            fitness = (fitness - fitness[np.argmin(fitness, axis=0), :]) / (
                fitness[np.argmax(fitness, axis=0), :]
                - fitness[np.argmin(fitness, axis=0), :]
            )
        return fitness

    def best_genotype(self, population):
        best = np.argmin(self.calculate_fitness(population), axis=0)
        vals = {
            key: value[best, :]
            for key, value in self.chromosomes_to_values(population).items()
        }
        return self.calculate_fitness(population)[best, :], population[best, :], vals


if __name__ == "__main__":
    test = np.random.randint(2, size=(100, 10), dtype="I")

    ff = FitnessFunction({"x": (10, -5, 5)}, lambda v: v["x"])
    fitness = ff.calculate_fitness(test)
    encoded_values = chromosomes_to_value(test, 0, 10, -5, 5)
    assert np.array_equal(fitness, encoded_values)

    assert x_square({"x": 2}) == 4
    assert himmelblau({"x":3,"y":2}) == 0
