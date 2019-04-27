import random
from StringVersion import individual


class OnePoint:
    def __init__(self, probability=0.01):
        self.mutation_probability = probability

    def run(self, population):
        if self.mutation_probability == 0.00:
            return population

        mutated_population, size = {}, len(population)
        for i in range(0, size):
            new_individual = {}
            for key, value in population[i].genotype().items():
                new_value = ""
                for index, char in enumerate(value):
                    if index > 3 and random.random() < self.mutation_probability:
                        char = "0" if char == "1" else "1"
                        new_value = new_value + char
                    else:
                        new_value = new_value + char
                new_individual[key] = new_value
            new_individual = individual.Individual(new_individual)
            mutated_population[i] = new_individual
        return mutated_population
