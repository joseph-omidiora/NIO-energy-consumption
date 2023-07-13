import numpy as np
import firefly

# Define the Griewank function
def griewank(x):
    term1 = np.sum(x ** 2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1))))
    return 1 + term1 - term2


# Define the search space and dimension
lb = np.array([-600] * 10)
ub = np.array([600] *10)
dim = 10


# Define the optimal value
optimal_value = 0.0

# Define the number of fireflies and maximum iteration
population = 10 * dim
max_iter = 100 * dim

# Run the Firefly Algorithm
best_solution, best_fitness = firefly.firefly_algorithm(griewank, lb, ub, dim, population, max_iter, optimal_value)


# Print the result
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
