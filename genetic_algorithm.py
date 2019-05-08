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
        # trace is used to store results for every iteration
        self._trace = {}
        # determines whether or not show messages with progress and time measurement
        self._show_messages = show_messages

    def run(self, iterations=100):
        if self._show_messages:
            print("Starting")
        t1 = time()

        # restart trace, initialize population and calculate fitness
        self._trace = {}
        population = self._initializer.initialize_population()
        fitness = self._fitness.calculate_fitness(population)

        # perform selection, crossover and mutation and calculate fitness of the new population
        for i in range(0, iterations):
            selected = self._selection.select(population, fitness)
            crossed = self._crossover.crossover(selected)
            mutated = self._mutation.mutate(crossed)
            population = mutated
            fitness = self._fitness.calculate_fitness(population)

            # save results
            results = self._fitness.results()
            self._trace[i] = {
                "solution": results[0],
                "best fitness": results[1],
                "mean fitness": results[2],
                "worst fitness": results[3],
            }

            # if showing messages, calculate algorithms progress
            if self._show_messages and 10 * i % iterations == 0:
                print(f"{100*i/iterations+10}% done!")

        final_solution = self._trace[0]
        for key, value in self._trace.items():
            if value["best fitness"] < final_solution["best fitness"]:
                final_solution = value

        if self._show_messages:
            print(f"Time elapsed: {time()-t1}")

        return final_solution

    def best_solution_history(self):
        iterations = range(0, len(self._trace))
        best_values = []
        for key, value in self._trace.items():
            best_values.append(value["best fitness"])
        return np.reshape(np.array(best_values), (len(best_values),)), np.array(iterations)

    def mean_solution_history(self):
        iterations = range(0, len(self._trace))
        mean_values = []
        for key, value in self._trace.items():
            mean_values.append(value["mean fitness"])
        return np.reshape(np.array(mean_values), (len(mean_values),)), np.array(iterations)

    def worst_solution_history(self):
        iterations = range(0, len(self._trace))
        worst_values = []
        for key, value in self._trace.items():
            worst_values.append(value["worst fitness"])
        return np.reshape(np.array(worst_values), (len(worst_values),)), np.array(iterations)


if __name__ == "__main__":
    x_sq_encoding = OrderedDict({"x": (10, -10, 10)})
    ga = GeneticAlgorithm(
        x_sq_encoding, x_square, 100, tournament_size=5, crossover_probability=0.9
    )
    solution = ga.run(100)
    print(solution)

    assert solution["solution"]["x"] < 0.1
    assert solution["best fitness"] < 0.1

    h_encoding = OrderedDict({"x": (10, -5, 5), "y": (10, -5, 5)})
    ga = GeneticAlgorithm(
        h_encoding,
        himmelblau,
        1000,
        tournament_size=10,
        crossover_probability=0.9,
        mutation_probability=0.2,
    )
    solution = ga.run(100)
    print(solution)

    assert solution["best fitness"] < 0.1
