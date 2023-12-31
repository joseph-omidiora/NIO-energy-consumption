import numpy as np
import wolf_optimizer

# Define the benchmark function (e.g., Rastrigin function)
def rastrigin(x):
    return 10*len(x) + sum([(xi**2 - 10*np.cos(2*np.pi*xi)) for xi in x])

# Define the search space and dimension
search_space = [-5.12, 5.12] # search space for each dimension
dim = 10 # dimension of the search space

# Define the population size and maximum number of iterations
population_size = 10 *dim 
max_iterations = 100 * dim

# Define the optimal value (i.e., the minimum value of the benchmark function)
optimal_value = 0

# Call the GWO function to solve the benchmark function
wolf_optimizer.gwo(rastrigin, search_space, dim, population_size, max_iterations, optimal_value)
