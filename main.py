from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from fitness_function import himmelblau

h_encoding = OrderedDict({"x": (32, -5, 5), "y": (32, -5, 5)})
ga = GeneticAlgorithm(
    h_encoding,
    himmelblau,
    200,
    tournament_size=2,
    crossover_probability=0.9,
    crossover_type="1p1v",
    mutation_probability=0.01,
    show_messages=True
)
results = ga.run(100)

print(f"Best fitness: {results['fitness']} for values: {results['values']}")

x, y = ga.avg_value_history()
plt.plot(x, y)
plt.title("Mean value from all population")
plt.show()

x, y = ga.best_value_history()
ylog = np.log10(y)
plt.plot(x,ylog)
plt.title("Log10(best value)")
plt.show()

