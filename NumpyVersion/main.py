from NumpyVersion.genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from NumpyVersion.fitness_function import himmelblau

"""
TODO:
why avg fitness is wrong???
some nice graphs to show results
"""

h_encoding = OrderedDict({"x": (20, -5, 5), "y": (20, -5, 5)})
ga = GeneticAlgorithm(
    h_encoding,
    himmelblau,
    2000,
    40,
    tournament_size=2,
    crossover_probability=0.8,
    mutation_probability=0,
)
results = ga.run(100)

print(results)

x, y = ga.avg_value_history()
y = np.log(y)
plt.plot(x, y)
plt.show()
