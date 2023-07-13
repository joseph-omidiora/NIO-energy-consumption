import numpy as np
import cuckoo_func

# Define the Rosenbrock function
def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

# Define the search space and dimension
lb = -5.12
ub = 5.12
dim = 10 # dimension of the search space

# Define the population size and maximum number of iterations
population_size = 10 *dim 
max_iterations = 100 * dim

# Define the optimal value (i.e., the minimum value of the benchmark function)
optimal_value = 1e-6

# Call the cuckoo function to solve the benchmark function
best_solution, best_fitness = cuckoo_func.cuckoo_search(rosenbrock, dim, population_size, max_iterations, lb, ub)
print("Test case: Rosenbrock function")
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)