import numpy as np
import bat_algorithm

def sphere(x):
    return np.sum(x**2)

dim = 10 
bat_alg = bat_algorithm.BatAlgorithm(sphere,dim)
best_fitness, best_location = bat_alg.optimize()

print('Best fitness:', best_fitness)
print('Best location:', best_location)