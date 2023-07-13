import numpy as np

# Define the cuckoo search algorithm# Define the cuckoo search algorithm
def cuckoo_search(fitness_func, dim, population_size, iterations, lb, ub, pa=0.25):
    # Initialize the population
    population = np.random.uniform(lb, ub, (population_size, dim))
    fitness = np.array([fitness_func(p) for p in population])

    # Store the best solution found so far
    best_fitness_idx = np.argmin(fitness)
    best_fitness = fitness[best_fitness_idx]
    best_solution = population[best_fitness_idx]

    # Main loop
    for t in range(iterations):
        # Generate new solutions using Levy flights
        new_population = np.empty((population_size, dim))
        for i in range(population_size):
            # Choose a random solution to mimic
            j = np.random.randint(population_size)
            while j == i:
                j = np.random.randint(population_size)
            # Generate a new solution using Levy flights
            beta = 1.5
            s = np.random.normal(0, 1, dim)
            step = np.power(np.fabs(s), 1.0 / beta)
            levy = np.random.normal(0, 1, dim) * step
            new_solution = population[i] + 0.01 * levy * (population[i] - population[j])
            new_solution = np.clip(new_solution, lb, ub)
            new_fitness = fitness_func(new_solution)
            # Evaluate the fitness of the new solution
            if new_fitness < fitness[i]:
                fitness[i] = new_fitness
                population[i] = new_solution
            # Abandon a fraction of solutions and generate new ones randomly
            if np.random.rand() < pa:
                population[i] = np.random.uniform(lb, ub, dim)
                fitness[i] = fitness_func(population[i])
        # Update the best solution found so far
        new_best_fitness_idx = np.argmin(fitness)
        new_best_fitness = fitness[new_best_fitness_idx]
        new_best_solution = population[new_best_fitness_idx]
        if new_best_fitness < best_fitness:
            best_fitness = new_best_fitness
            best_solution = new_best_solution
        print("Iteration:", t+1, "Best fitness:", best_fitness)
    return best_solution, best_fitness


