import numpy as np

def gwo(f, search_space, dim, population_size, max_iterations, optimal_value):
    # Initialize the positions of the wolves
    positions = np.random.uniform(search_space[0], search_space[1], (population_size, dim))
    
    # Initialize the alpha, beta, and delta wolves
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")
    beta_pos = np.zeros(dim)
    beta_score = float("inf")
    delta_pos = np.zeros(dim)
    delta_score = float("inf")
    
    # Main loop
    for iteration in range(max_iterations):
        # Update the positions of the alpha, beta, and delta wolves
        for i in range(population_size):
            # Calculate the fitness of the current wolf
            fitness = f(positions[i])
            
            # Update alpha, beta, and delta
            if fitness < alpha_score:
                delta_pos = beta_pos.copy()
                delta_score = beta_score
                beta_pos = alpha_pos.copy()
                beta_score = alpha_score
                alpha_pos = positions[i].copy()
                alpha_score = fitness
            elif fitness < beta_score:
                delta_pos = beta_pos.copy()
                delta_score = beta_score
                beta_pos = positions[i].copy()
                beta_score = fitness
            elif fitness < delta_score:
                delta_pos = positions[i].copy()
                delta_score = fitness
        
        # Update the positions of the wolves
        a = 2.0 - 2.0 * (iteration / max_iterations) # linearly decreasing from 2 to 0
        for i in range(population_size):
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            A1 = a * (2.0 * r1 - 1.0)
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * alpha_pos - positions[i])
            X1 = alpha_pos - A1 * D_alpha
            
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            A2 = a * (2.0 * r1 - 1.0)
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * beta_pos - positions[i])
            X2 = beta_pos - A2 * D_beta
            
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            A3 = a * (2.0 * r1 - 1.0)
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * delta_pos - positions[i])
            X3 = delta_pos - A3 * D_delta
            
            positions[i] = (X1 + X2 + X3) / 3.0
        
        # Print the best fitness value found so far every 10 iterations
        if (iteration+1) % 1 == 0:
            print(f"Iteration {iteration+1}/{max_iterations}: Best fitness = {alpha_score}")
    
    # Check for convergence
    if alpha_score <= optimal_value:
        print(f"Solution found: {alpha_pos}")
        return alpha_pos
    else:
        print(f"Solution not found after {max_iterations} iterations")
        return None
