import numpy as np
import bat_algorithm


# Define the Rastrigin function
def rastrigin(x):
    A = 10
    return A * dim + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


dim = 10 
bat_alg = bat_algorithm.BatAlgorithm(rastrigin,dim)
best_fitness, best_location = bat_alg.optimize()

print('Best fitness:', best_fitness)
print('Best location:', best_location)