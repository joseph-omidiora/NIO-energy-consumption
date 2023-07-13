import numpy as np
import firefly


# Define the Rastrigin function
def rastrigin(x):
    A = 10
    return A * dim + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


# Define the search space and dimension
lb = np.array([-5.12] * 10)
ub = np.array([5.12] *10)
dim = 10


# Define the optimal value
optimal_value = 0.0

# Define the number of fireflies and maximum iteration
population = 10 * dim
max_iter = 100 * dim

# Run the Firefly Algorithm
best_solution, best_fitness = firefly.firefly_algorithm(rastrigin, lb, ub, dim, population, max_iter, optimal_value)


# Print the result
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
