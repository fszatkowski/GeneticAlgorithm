import initialization
import selection
import crossover
import mutation


class GeneticAlgorithm:
    def __init__(
        self,
        variables=None,
        fitness_function=None,
        initialization_type=None,
        population_size=100,
        selection_type=None,
        tournament_size=20,
        crossover_type=None,
        crossover_probability=1.0,
        mutation_type=None,
        mutation_probability=0.01,
    ):
        # we need the type of individual
        self.variables = variables

        # fitness function takes an individual and returns one number
        # it has to be defined
        self.fitness_function = fitness_function

        # initialization function returns indivduals
        if initialization_type == "uniform":
            self.initializer = initialization.Uniform(self.variables)
        self.population_size = population_size

        # selection function returns list pairs of individuals selected for mating
        if selection_type == "tournament":
            self.selection = selection.Tournament(
                tournament_size, self.fitness_function
            )

        # crossover function has its own probablity, takes two old indivduals and returns two new individuals
        if crossover_type == "one_point":
            self.crossover = crossover.OnePoint(crossover_probability)

        if mutation_type == "one_point":
            self.mutation = mutation.OnePoint(mutation_probability)

    def run(self, iterations=100):
        population = self.initializer.run(self.population_size)
        perc = 1
        for i in range(0, iterations):
            selected_population = self.selection.run(population)
            crossed_pop = self.crossover.run(selected_population)
            mutated_pop = self.mutation.run(crossed_pop)
            population = mutated_pop
            progress = i / iterations
            if progress > perc / 20:
                print(f"{perc*5}% done!")
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


# lets try Himmelblau's function with restricted search space
def fitness_function(individual):
    x, y = individual.fenotype()["x"], individual.fenotype()["y"]
    return pow(x * x + y - 11, 2) + pow(x + y * y - 7, 2)


variables = {"x": (-1250, 1250), "y": (-1250, 1250)}

ga = GeneticAlgorithm(
    variables=variables,
    fitness_function=fitness_function,
    initialization_type="uniform",
    population_size=1000,
    selection_type="tournament",
    tournament_size=50,
    crossover_type="one_point",
    crossover_probability=0.8,
    mutation_type="one_point",
    mutation_probability=0.01,
)
ga.run(200)
