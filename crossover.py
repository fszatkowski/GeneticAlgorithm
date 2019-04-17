import individual
import random


class OnePoint:
    def __init__(self, probability):
        self.prob = probability

    def run(self, selected):
        crossed = {}
        # iterate over pairs in selected population
        for i in range(0, int(len(selected) / 2)):
            # determine whether or not do crossover
            do_crossover = random.uniform(0, 1)
            if do_crossover > self.prob:
                crossed[2 * i] = selected[2 * i]
                crossed[2 * i + 1] = selected[2 * i + 1]
            else:
                # if we do crossover, then iterate over every key and value in both individuals and set new value
                new_individual1 = {}
                new_individual2 = {}
                for key, value1 in selected[2 * i].genotype().items():
                    value2 = selected[2 * i + 1].genotype()[key]
                    point = random.randint(0, 32)
                    new_value1 = value1[0:point] + value2[point:len(value2)]
                    new_value2 = value2[0:point] + value1[point:len(value1)]
                    new_individual1[key] = new_value1
                    new_individual2[key] = new_value2
                crossed[2 * i] = individual.Individual(new_individual1)
                crossed[2 * i + 1] = individual.Individual(new_individual2)
        return crossed
