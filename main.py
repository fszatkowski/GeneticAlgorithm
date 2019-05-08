from genetic_algorithm import GeneticAlgorithm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import OrderedDict
from fitness_function import x_square

h_encoding = OrderedDict({"x": (4, -10, 10)})
ga = GeneticAlgorithm(
    h_encoding,
    x_square,
    100,
    tournament_size=2,
    crossover_probability=0.8,
    crossover_type="1p1v",
    mutation_probability=0.001,
    show_messages=True,
)
results = ga.run(100)

print(f"Best fitness: {results['best fitness']} for values: {results['solution']}")

best, iters = ga.best_solution_history()
mean, _ = ga.mean_solution_history()
worst, _ = ga.worst_solution_history()

# best, mean, worst = np.log10(best), np.log10(mean), np.log10(worst)

data = pd.DataFrame(best, index=iters, columns=["best value"])
data["mean value"] = pd.Series(mean)
data["worst value"] = pd.Series(worst)

sns.lineplot(data=data)
plt.legend()
plt.show()
