import random

from individual import Individual
from individual import float_to_bin

"""
__init__ takes encoding in form: 'x':(min, max)
then run can be used to return list of individuals matching given encoding
"""


class Uniform:
    def __init__(self, encoding):
        self.encoding = encoding

    def run(self, population_size):
        new_population = {}
        for i in range(0, population_size):
            new_population[i] = Individual(
                {
                    key: float_to_bin(random.uniform(value[0], value[1]))
                    for key, value in self.encoding.items()
                }
            )
        return new_population
