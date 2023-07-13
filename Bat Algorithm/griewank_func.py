import numpy as np
import bat_algorithm


# Define the Griewank function
def griewank(x):
    term1 = np.sum(x ** 2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1))))
    return 1 + term1 - term2

dim = 10 
bat_alg = bat_algorithm.BatAlgorithm(griewank,dim)
best_fitness, best_location = bat_alg.optimize()

print('Best fitness:', best_fitness)
print('Best location:', best_location)