import numpy as np
dim = 10
# Define the sphere function to be optimized
def sphere(x):
    return np.sum(x ** 2)

# Define the Griewank function
def griewank(x):
    term1 = np.sum(x ** 2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1))))
    return 1 + term1 - term2

# Define the Ackley function
def ackley(x):
    term1 = -0.2 * np.sqrt(np.mean(x ** 2))
    term2 = np.mean(np.cos(2 * np.pi * x))
    return -20 * np.exp(term1) - np.exp(term2) + 20 + np.e


# Define the Rosenbrock function
def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


# Define the Rastrigin function
def rastrigin(x):
    A = 10
    return A * dim + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

