import numpy as np
from benchmark_func import sphere

def bat_algorithm(benchmark_func.sphere, search_space, dim, max_iter, optimal_val, population, loudness=0.5, pulse_rate=0.5):
    """
    Bat Algorithm for optimization.

    Args:
        benchmark_func (function): The function to be optimized.
        search_space (ndarray): The search space.
        dim (int): The dimensionality of the search space.
        max_iter (int): The maximum number of iterations.
        optimal_val (float): The optimal value of the benchmark function.
        population (int): The number of bats in the population.
        loudness (float, optional): The loudness of the bats. Defaults to 0.5.
        pulse_rate (float, optional): The pulse rate of the bats. Defaults to 0.5.

    Returns:
        ndarray: The best solution found by the algorithm.
        float: The fitness value of the best solution found by the algorithm.
    """
    # Initialization
    bats = np.random.rand(population, dim) * (search_space[:, 1] - search_space[:, 0]) + search_space[:, 0]
    velocities = np.zeros((population, dim))
    fitness_values = np.apply_along_axis(benchmark_func, 1, bats)
    best_bat = bats[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)

    # Main loop
    for t in range(max_iter):
        # Update loudness and pulse rate
        loudness *= 0.99
        pulse_rate = 0.5 * (1 - np.exp(-0.5 * t))

        # Update bat positions and velocities
        for i in range(population):
            frequencies = np.random.normal(0, 1, dim)
            velocities[i] += (bats[i] - best_bat) * frequencies
            bats[i] += velocities[i]

            # Apply constraints
            bats[i] = np.clip(bats[i], search_space[:, 0], search_space[:, 1])

            # Generate a new solution with probability pulse_rate
            if np.random.rand() < pulse_rate:
                new_bat = best_bat + np.random.normal(0, 1, dim) * loudness
                new_bat = np.clip(new_bat, search_space[:, 0], search_space[:, 1])
                new_fitness = benchmark_func.sphere(new_bat)
                if new_fitness < best_fitness:
                    best_bat = new_bat
                    best_fitness = new_fitness

        # Display progress
        print(f"Iteration {t+1}/{max_iter} - Best Fitness: {best_fitness:.4f} - Optimal Fitness: {optimal_val:.4f}")

    return best_bat, best_fitness
