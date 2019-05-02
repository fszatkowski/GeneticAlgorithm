"""
Performs genetic algorithm with specifies parameters
variables passed as OrderedDict: {key:(bits, min_val, max_val)}
functions argument order must match dicts argument order
"""

from collections import OrderedDict
from time import time

import numpy as np

from crossover import OnePointForEveryVariable, OnePointForWholeGenotype
from fitness_function import FitnessFunction, himmelblau, x_square
from initialization import Uniform
from mutation import UniformMutation
from selection import Tournament


class GeneticAlgorithm:
    def __init__(
        self,
        variables_encoding,
        fitness,
        population_size,
        tournament_size=1,
        crossover_probability=0.9,
        crossover_type="1p1v",
        mutation_probability=0.01,
        show_messages=True,
    ):
        bit_sum = 0
        for key, value in variables_encoding.items():
            bit_sum += value[0]
        if population_size % 2 != 0 or population_size <= 0:
            raise ValueError("Population size must be positive multiple of 2")
        if tournament_size < 1 or tournament_size > population_size:
            raise ValueError(
                "Tournament size must be positive and cannot exceed population size"
            )
        if crossover_probability < 0 or crossover_probability > 1:
            raise ValueError("Crossover probability must be in range (0,1)")
        if mutation_probability < 0 or mutation_probability > 1:
            raise ValueError("Mutation probability must be in range(0,1)")

        self._initializer = Uniform((population_size, bit_sum))
        self._fitness = FitnessFunction(variables_encoding, fitness)
        self._selection = Tournament(tournament_size)
        if crossover_type == "1p1v":
            self._crossover = OnePointForEveryVariable(
                variables_encoding, probability=crossover_probability
            )
        elif crossover_type == "1pall":
            self._crossover = OnePointForWholeGenotype(
                probability=crossover_probability
            )
        self._mutation = UniformMutation(probability=mutation_probability)
        self._trace = {}
        self._show_messages = show_messages

    def run(self, iterations=100):
        self._trace = {}
        population = self._initializer.initialize_population()

        if self._show_messages:
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
            if self._show_messages and 10 * i % iterations == 0:
                print(f"{100*(i)/iterations+10}% done!")

        result = self._trace[0]
        for key, value in self._trace.items():
            if value["fitness"] < result["fitness"]:
                result = value

        if self._show_messages:
            print(f"Time elapsed: {time()-t1}")

        return result

    def best_value_history(self):
        iterations = range(0, len(self._trace))
        best_values = []
        for key, value in self._trace.items():
            best_values.append(value["fitness"])
        return np.array(iterations), np.reshape(np.array(best_values), (len(best_values),))

    def avg_value_history(self):
        iterations = range(0, len(self._trace))
        avg_values = []
        for key, value in self._trace.items():
            avg_values.append(value["avg_fitness"])
        return np.array(iterations), np.reshape(np.array(avg_values), (len(avg_values),))


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
