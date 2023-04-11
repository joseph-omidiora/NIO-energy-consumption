import numpy as np

import cuckoo

# Define the Ackley function
def ackley(x):
    term1 = -0.2 * np.sqrt(np.mean(x ** 2))
    term2 = np.mean(np.cos(2 * np.pi * x))
    return -20 * np.exp(term1) - np.exp(term2) + 20 + np.e


# Define the search space and dimension
search_space = [-30, 30] # search space for each dimension
dim = 10 # dimension of the search space

# Define the population size and maximum number of iterations
population_size = 10 *dim 
max_iterations = 100 * dim

# Define the optimal value (i.e., the minimum value of the benchmark function)
optimal_value = 0

# Call the GWO function to solve the benchmark function
cuckoo.cuckoo_search(ackley, search_space, dim, population_size, max_iterations, optimal_value)
