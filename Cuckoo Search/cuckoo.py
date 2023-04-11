import numpy as np
from math import gamma

def cuckoo_search(func, search_space, dim, n_population, n_iter, optimal_value):
    # func: benchmark function to optimize
    # search_space: search space boundaries for each dimension
    # dim: number of dimensions in the search space
    # n_population: number of cuckoos in the population
    # n_iter: maximum number of iterations to run
    # optimal_value: the known optimal value of the benchmark function
    
    # Generate initial population
    population = np.random.uniform(search_space[:, 0], search_space[:, 1], (n_population, dim))
    fitness = np.zeros(n_population)
    for i in range(n_population):
        fitness[i] = func(population[i])
    
    # Sort the population and select the best solutions
    population = population[np.argsort(fitness)]
    fitness = fitness[np.argsort(fitness)]
    best_solution = population[0]
    best_fitness = fitness[0]
    
    # Loop until maximum number of iterations is reached
    for iteration in range(n_iter):
        # Generate new solutions using Levy flights
        new_population = np.zeros((n_population, dim))
        beta = 3/2
        sigma = (np.abs(np.prod(search_space[:, 1] - search_space[:, 0]))**(1/dim)) / ((gamma(1 + beta) * np.sin(np.pi*beta/2)) / (gamma((1+beta)/2) * beta * 2**((beta-1)/2)))
        for i in range(n_population):
            s = np.random.randn(dim) * sigma
            levy = 1 / (1 + (np.linalg.norm(s)**2)**(beta/2))
            new_population[i] = population[i] + levy * s * 0.01
        
        # Evaluate fitness of new solutions
        new_fitness = np.zeros(n_population)
        for i in range(n_population):
            if np.all(new_population[i] >= search_space[:, 0]) and np.all(new_population[i] <= search_space[:, 1]):
                new_fitness[i] = func(new_population[i])
            else:
                new_fitness[i] = np.inf
        
        # Replace some solutions in the population with new ones
        idx = np.argsort(new_fitness)
        new_population = new_population[idx]
        new_fitness = new_fitness[idx]
        for i in range(int(n_population/2)):
            j = np.random.randint(0, n_population)
            if new_fitness[j] < fitness[i]:
                population[i] = new_population[j]
                fitness[i] = new_fitness[j]
        
        # Sort the population and select the best solutions
        population = population[np.argsort(fitness)]
        fitness = fitness[np.argsort(fitness)]
        if fitness[0] < best_fitness:
            best_solution = population[0]
            best_fitness = fitness[0]
        
        # Check if optimal value has been reached
        if np.abs(best_fitness - optimal_value) < 1e-6:
            break
    
    return best_solution, best_fitness
