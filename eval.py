from genetic_algorithm import GeneticAlgorithm
from fitness_function import himmelblau
import pandas as pd
from collections import Iterable


class Evaluator:
    def __init__(self, variables, fitness, tests=1, avg_tests=False, show_messages=True):
        self._variables_encoding = variables
        self._fitness_function = fitness
        self._show_messages = show_messages
        self._tests = tests
        self._avg_tests = avg_tests

    def iterations_population(
        self,
        population_size=100,
        iterations=100,
        tour_size=1,
        cross_prob=0.9,
        cross_type="1p1v",
        mutation_prob=0.01,
    ):
        if not isinstance(population_size, Iterable) or not isinstance(
            iterations, Iterable
        ):
            raise ValueError("Population size and iteration should be iterable")

        operations = len(population_size) * len(iterations) * self._tests

        results = pd.DataFrame(columns=population_size)
        for row in iterations:
            for column in population_size:
                if self._show_messages:
                    print(f"Calcucating: population {column}, iterations {row}")
                ga = GeneticAlgorithm(
                    self._variables_encoding,
                    self._fitness_function,
                    column,
                    tournament_size=tour_size,
                    crossover_probability=cross_prob,
                    crossover_type=cross_type,
                    mutation_probability=mutation_prob,
                    show_messages=False,
                )
                test_result = ""
                if self._avg_tests:
                    mean_variables = {"fitness":0}
                    for var in self._variables_encoding:
                        mean_variables[var] = 0
                    for i in range(0,self._tests):
                        val = ga.run(row)
                        mean_variables["fitness"] += val["fitness"]
                        for key, value in val["values"].items():
                            mean_variables[key] += value
                    for key, value in mean_variables.items():
                        mean_value = value/self._tests
                        mean_variables[key] = mean_value
                        test_result += f"{key}: {mean_value} "
                else:
                    best_variables = {}
                    for i in range(0, self._tests):
                        val = ga.run(row)
                        if "fitness" not in best_variables:
                            best_variables["fitness"] = val["fitness"]
                            for key, value in val["values"].items():
                                best_variables[key] = value
                        else:
                            if val["fitness"] < best_variables["fitness"]:
                                best_variables["fitness"] = val["fitness"]
                                for key, value in val["values"].items():
                                    best_variables[key] = value
                    for key, value in best_variables.items():
                        test_result += f"{key}: {value} "
                results.loc[row, column] = test_result
        return results


v_encoding = {"x":(10,-5,5), "y":(10,-5,5)}
evaluator = Evaluator(v_encoding, himmelblau, tests=4, avg_tests=False, show_messages=True)
res = evaluator.iterations_population(population_size=[10, 20, 50, 100, 200], iterations=[100, 200])
print(res)