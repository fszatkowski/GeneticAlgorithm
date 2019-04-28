import random


""" 
performs tournament selection with the number of tournaments specified by tournaments parameter
"""


class Tournament:
    def __init__(self, tournaments, fitness_function):
        if tournaments < 1:
            tournaments = 1
        self.tour_size = tournaments
        self.fitness_function = fitness_function

    def run(self, population):
        size, selected = len(population), {}

        # do len(population) tournaments to determine new population
        for i in range(0, size):
            candidates = [
                random.randint(0, size - 1) for i in range(0, int(self.tour_size))
            ]
            winner = candidates[0]
            for c in candidates:
                if self.fitness_function(population[c]) < self.fitness_function(
                    population[winner]
                ):
                    winner = c
            selected[i] = population[winner]
        return selected
