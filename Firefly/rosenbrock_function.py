import numpy as np
import firefly

# Define the Rosenbrock function
def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


# Define the search space and dimension
lb = np.array([-2.048] * 10)
ub = np.array([2.048] *10)
dim = 10


# Define the optimal value
optimal_value = 0.0

# Define the number of fireflies and maximum iteration
population = 10 * dim
max_iter = 100 * dim

# Run the Firefly Algorithm
best_solution, best_fitness = firefly.firefly_algorithm(rosenbrock, lb, ub, dim, population, max_iter, optimal_value)


# Print the result
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
