import numpy as np
import wolf_optimizer

# Define the Griewank function
def griewank(x):
    term1 = np.sum(x ** 2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1))))
    return 1 + term1 - term2

# Define the search space and dimension
search_space = [-600, 600] # search space for each dimension
dim = 10 # dimension of the search space

# Define the population size and maximum number of iterations
population_size = 10 *dim 
max_iterations = 100 * dim

# Define the optimal value (i.e., the minimum value of the benchmark function)
optimal_value = 0

# Call the GWO function to solve the benchmark function
wolf_optimizer.gwo(griewank, search_space, dim, population_size, max_iterations, optimal_value)

