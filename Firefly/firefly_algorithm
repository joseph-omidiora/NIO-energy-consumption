import numpy as np

def firefly_algorithm(fun, lb, ub, dim, n_fireflies, max_iter, optimal_value):
    # Initialization
    alpha = 0.2  # Light absorption coefficient
    gamma = 1.0  # Attraction coefficient
    beta_min = 0.2  # Minimum value of beta
    beta_max = 1.0  # Maximum value of beta

    # Initialize fireflies
    fireflies = np.random.uniform(lb, ub, (n_fireflies, dim))

    # Main loop
    for t in range(max_iter):
        # Evaluate fitness
        fitness = np.array([fun(f) for f in fireflies])

        # Sort fireflies by fitness (ascending order)
        idx = np.argsort(fitness)
        fireflies = fireflies[idx]

        # Update the light intensity
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] < fitness[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta_min + (beta_max - beta_min) * np.exp(-gamma * r ** 2)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * np.random.randn(dim)

        # Clip fireflies to the search space
        fireflies = np.clip(fireflies, lb, ub)

        # Print current status
        best_fitness = fitness[0]
        print("Iteration {}: Best Fitness = {:.6f}".format(t + 1, best_fitness))

        # Check if optimal value is reached
        if abs(best_fitness - optimal_value) < 1e-6:
            break

    # Return the best solution
    return fireflies[0], best_fitness
