import numpy as np
import wolf_optimizer

# Define the sphere function to be optimized
def sphere(x):
    return np.sum(x ** 2)

# Define the search space and dimension
search_space = [-5.12, 5.12] # search space for each dimension
dim = 10 # dimension of the search space

# Define the population size and maximum number of iterations
population_size = 10 *dim 
max_iterations = 100 * dim

# Define the optimal value (i.e., the minimum value of the benchmark function)
optimal_value = 1e-6

# Call the GWO function to solve the benchmark function
wolf_optimizer.gwo(sphere, search_space, dim, population_size, max_iterations, optimal_value)
