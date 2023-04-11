import numpy as np
import firefly


# Define the Ackley function
def ackley(x):
    term1 = -0.2 * np.sqrt(np.mean(x ** 2))
    term2 = np.mean(np.cos(2 * np.pi * x))
    return -20 * np.exp(term1) - np.exp(term2) + 20 + np.e



# Define the search space and dimension
lb = np.array([-30] * 20)
ub = np.array([30] *20)
dim = 20


# Define the optimal value
optimal_value = 0.0

# Define the number of fireflies and maximum iteration
population = 10 * dim
max_iter = 100 * dim

# Run the Firefly Algorithm
best_solution, best_fitness = firefly.firefly_algorithm(ackley, lb, ub, dim, population, max_iter, optimal_value)


# Print the result
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
