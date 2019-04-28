from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from fitness_function import himmelblau

"""
TODO:
why avg fitness is wrong???
some nice graphs to show results
"""

h_encoding = OrderedDict({"x": (20, -5, 5), "y": (20, -5, 5)})
ga = GeneticAlgorithm(
    h_encoding,
    himmelblau,
    1000,
    40,
    tournament_size=100,
    crossover_probability=0.8,
    mutation_probability=0.05,
)
results = ga.run(100)

print(f"Best fitness: {results['fitness']} for values: {results['values']}")

x, y = ga.avg_value_history()
plt.plot(x, y)
plt.title("Mean value from all population")
plt.show()

x, y = ga.best_value_history()
y = np.log10(y)
plt.plot(x,y)
plt.title("Log10(best value)")
plt.show()