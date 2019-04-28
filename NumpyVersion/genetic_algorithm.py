"""
Performs genetic algorithm with specifies parameters
variables passed as OrderedDict: {key:(bits, min_val, max_val)}
functions argument order must match dicts argument order
"""

from collections import OrderedDict
from time import time

import numpy as np

from NumpyVersion.crossover import OnePoint
from NumpyVersion.fitness_function import FitnessFunction, himmelblau, x_square
from NumpyVersion.initialization import Uniform
from NumpyVersion.mutation import UniformMutation
from NumpyVersion.selection import Tournament


class GeneticAlgorithm:
    def __init__(
        self,
        variables_encoding,
        fitness,
        population_size,
        chromosome_length,
        tournament_size=1,
        crossover_probability=1,
        mutation_probability=0.01,
    ):
        bit_sum = 0
        for key, value in variables_encoding.items():
            bit_sum += value[0]
        if bit_sum != chromosome_length:
            raise ValueError(
                "Chromosome length does not match the number of bits in passed variables"
            )

        if population_size % 2 != 0:
            raise ValueError("Population size must be multiple of 2")

        self._initializer = Uniform((population_size, chromosome_length))
        self._fitness = FitnessFunction(variables_encoding, fitness)
        self._selection = Tournament(tournament_size)
        self._crossover = OnePoint(
            variables_encoding, probability=crossover_probability
        )
        self._mutation = UniformMutation(probability=mutation_probability)
        self._trace = {}

    def run(self, iterations=100):
        population = self._initializer.initialize_population()

        print("Starting")
        t1 = time()

        for i in range(0, iterations):
            fitness = self._fitness.calculate_fitness(population)
            selected = self._selection.select(population, fitness)
            crossed = self._crossover.crossover(selected)
            mutated = self._mutation.mutate(crossed)
            population = mutated
            self._trace[i] = {
                "values": self._fitness.best_genotype(population)[2],
                "fitness": self._fitness.best_genotype(population)[0],
                "avg_fitness": fitness.mean(),
            }
            if 10*i%iterations == 0:
                print(f"{100*(i+10)/iterations}% done!")

        result = self._trace[0]
        for key, value in self._trace.items():
            if value["fitness"] < result["fitness"]:
                result = value

        print(f"Time elapsed: {time()-t1}")

        return result

    def best_value_history(self):
        iterations = range(0, len(self._trace))
        best_values = []
        for key, value in self._trace.items():
            best_values.append(value["fitness"])
        return np.array(iterations), np.reshape(np.array(best_values), (100,))

    def avg_value_history(self):
        iterations = range(0, len(self._trace))
        avg_values = []
        for key, value in self._trace.items():
            avg_values.append(value["avg_fitness"])
        return np.array(iterations), np.reshape(np.array(avg_values), (100,))


if __name__ == "__main__":
    x_sq_encoding = OrderedDict({"x": (10, -10, 10)})
    ga = GeneticAlgorithm(
        x_sq_encoding, x_square, 100, 10, tournament_size=5, crossover_probability=0.9
    )
    t1 = time()
    results = ga.run(100)
    print(results)
    print(f"Time elapsed: {time()-t1}")

    assert results["values"]["x"] < 0.1
    assert results["fitness"] < 0.1

    h_encoding = OrderedDict({"x": (10, -5, 5), "y": (10, -5, 5)})
    ga = GeneticAlgorithm(
        h_encoding,
        himmelblau,
        1000,
        20,
        tournament_size=10,
        crossover_probability=0.9,
        mutation_probability=0.2,
    )
    t1 = time()
    results = ga.run(100)
    print(results)
    print(f"Time elapsed: {time()-t1}")

    assert results["fitness"] < 0.1
