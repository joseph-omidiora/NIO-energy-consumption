import numpy as np
import bat_algorithm

# Define the Rosenbrock function
def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


dim = 10 
bat_alg = bat_algorithm.BatAlgorithm(rosenbrock,dim)
best_fitness, best_location = bat_alg.optimize()

print('Best fitness:', best_fitness)
print('Best location:', best_location)