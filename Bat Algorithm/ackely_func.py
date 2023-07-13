import numpy as np
import bat_algorithm
# Define the sphere function to be optimized

def ackley(x):
    term1 = -0.2 * np.sqrt(np.mean(x ** 2))
    term2 = np.mean(np.cos(2 * np.pi * x))
    return -20 * np.exp(term1) - np.exp(term2) + 20 + np.e

dim = 10 
bat_alg = bat_algorithm.BatAlgorithm(ackley,dim)
best_fitness, best_location = bat_alg.optimize()

print('Best fitness:', best_fitness)
print('Best location:', best_location)