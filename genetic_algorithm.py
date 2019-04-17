"""
TODO:
-f. celu
-osobnik
-inicjalizacja
-kalkulacja funkcji celu
-selekcja
-crossover
-mutacja
-ewaluacja
"""

import initialization
import selection
import crossover


class GeneticAlgorithm:
    def __init__(
        self,
        encoding=None,
        fitness_function=None,
        initialization_type=None,
        population_size=100,
        selection_type=None,
        tournament_size=20,
        crossover_type=None,
        crossover_probability=1.0,
        mutation=None,
    ):
        # we need the type of individual
        self.encoding = encoding

        # fitness function takes an individual and returns one number
        # it has to be defined
        self.fitness_function = fitness_function

        # initialization function returns indivduals
        if initialization_type == "uniform":
            self.initializer = initialization.Uniform(self.encoding)
        self.population_size = population_size

        # selection function returns list pairs of individuals selected for mating
        if selection_type == "tournament":
            self.selection = selection.Tournament(
                tournament_size, self.fitness_function
            )

        # crossover function has its own probablity, takes two old indivduals and returns two new individuals
        if crossover_type == "1p":
            self.crossover = crossover.OnePoint(crossover_probability)

        self.mutation = mutation

    def run(self, iterations=100):
        population = self.initializer.run(self.population_size)
        perc = 1
        for i in range(0, iterations):
            selected_population = self.selection.run(population)
            crossed_pop = self.crossover.run(selected_population)
            population = crossed_pop
            progress = i/iterations
            if progress>perc/10:
                print(f"{perc*10}% done!")
                perc += 1

        best_individual = 0
        for i in range(0, len(population)):
            if fitness_function(population[i]) < fitness_function(
                population[best_individual]
            ):
                best_individual = i

        print(
            f"Result: {population[best_individual].fenotype()}, fitness function: {fitness_function(population[best_individual])}"
        )


# lets try Himmelblau's function and restrict it's search space
def fitness_function(individual):
    x = individual.fenotype()["x"]
    y = individual.fenotype()["y"]
    return pow(x * x + y - 11, 2) + pow(x + y * y - 7, 2)


encoding = {"x": (-5, 5), "y": (-5, 5)}

ga = GeneticAlgorithm(
    encoding=encoding,
    fitness_function=fitness_function,
    initialization_type="uniform",
    population_size=200,
    selection_type="tournament",
    tournament_size=50,
    crossover_type="1p",
    crossover_probability=0.8
)
ga.run(200)
