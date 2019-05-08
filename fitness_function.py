from collections import OrderedDict

import numpy as np


# calculates real value in range (min_value, max_value)
# from bits (first_bit, last_bit) in the chromosome
def decode_single_value(chromosomes, first_bit, last_bit, min_value, max_value):
    # get part of the matrix with bits representing value
    value_representation = chromosomes[:, first_bit:last_bit]
    L = value_representation.shape[1]
    # calculate bit weights mask
    bit_weights = np.zeros(shape=(1, L), dtype="I")
    for i in range(0, L):
        bit_weights[0, i] = pow(2, L - i - 1)
    # elementwise multiply bit mask and bits
    float_values = np.multiply(value_representation.astype("float"), bit_weights)
    # sum results for every single individual
    float_values = np.sum(float_values, axis=1, keepdims=True)
    # normalize results to (min, max)
    float_values = (
        min_value + ((max_value - min_value) / (pow(2, L) - 1)) * float_values
    )
    return float_values


# two test optimization functions for one and two variables
def x_square(v):
    return v["x"] * v["x"]


def himmelblau(v):
    return (v["x"] ** 2 + v["y"] - 11) ** 2 + (v["x"] + v["y"] ** 2 - 7) ** 2


class FitnessFunction:
    """
    variables - encoded variables in form of OrderedDict
    function - function object representing fitness function
    linear_scaling - if true, fitness is normalized to (0,1)
    """

    def __init__(self, variables, function, linear_scaling=False):
        self._variables = OrderedDict(variables)
        self._fitness_function = function
        self._scale = linear_scaling
        self._current_fitness = None
        self._current_solution = None

    # transforms bit matrix into dict {"value name": array with real values}
    def decode_chromosomes(self, population):
        encoded_values = OrderedDict()
        current_bit = 0
        for key, value in self._variables.items():
            encoded_values[key] = decode_single_value(
                population, current_bit, current_bit + value[0], value[1], value[2]
            )
            current_bit += value[0]
        return encoded_values

    # return fitness value for the whole population and calculate best solution
    def calculate_fitness(self, population):
        # decode chromosomes into real values and calculate fitness for them
        variables = self.decode_chromosomes(population)
        fitness = self._fitness_function(variables)
        # scale values to (0-1) if function called with scaling
        if self._scale:
            fitness = (fitness - fitness[np.argmin(fitness, axis=0), :]) / (
                fitness[np.argmax(fitness, axis=0), :]
                - fitness[np.argmin(fitness, axis=0), :]
            )

        self._current_fitness = fitness
        best_solution = {
            key: value[0, 0]
            for key, value in self.decode_chromosomes(
                population[np.argmin(fitness, axis=0), :]
            ).items()
        }
        self._current_solution = best_solution
        return fitness

    # return best solution and best, avg and worst fitness values
    def results(self):
        fitness = self._current_fitness
        best_solution = np.argmin(fitness, axis=0)
        worst_solution = np.argmax(fitness, axis=0)
        best_fitness = fitness[best_solution, :][0, 0]
        worst_fitness = fitness[worst_solution, :][0, 0]
        mean_fitness = fitness.mean()
        return self._current_solution, best_fitness, mean_fitness, worst_fitness


if __name__ == "__main__":
    test = np.random.randint(2, size=(100, 10), dtype="I")

    ff = FitnessFunction({"x": (10, -5, 5)}, lambda v: v["x"])
    fitness_values = ff.calculate_fitness(test)
    print(ff.results())
    encoding_result = decode_single_value(test, 0, 10, -5, 5)

    assert np.array_equal(fitness_values, encoding_result)
    assert x_square({"x": 2}) == 4
    assert himmelblau({"x": 3, "y": 2}) == 0
