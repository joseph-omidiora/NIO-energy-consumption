import numpy as np

def ACO(obj_func, dim, pop_size, max_iter, lb, ub, opt_val, rho, alpha, beta):
    
    # Initialize pheromone matrix
    tau = np.ones((pop_size, dim)) / (dim * pop_size)
    
    # Initialize best solution and best fitness
    best_sol = np.zeros(dim)
    best_fit = np.inf
    
    # Initialize ant population
    ants = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Start iterations
    for iter in range(max_iter):
        
        # Evaluate fitness of all ants
        fitness = np.array([obj_func(ant) for ant in ants])
        
        # Update best solution
        if np.min(fitness) < best_fit:
            best_sol = ants[np.argmin(fitness)]
            best_fit = np.min(fitness)
        
        # Update pheromone matrix
        delta_tau = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for j in range(dim):
                delta_tau[i, j] = 1 / (fitness[i] - opt_val)
        tau = (1 - rho) * tau + rho * delta_tau
        
        # Generate new ant population
        new_ants = np.zeros((pop_size, dim))
        for i in range(pop_size):
            ant = np.zeros(dim)
            for j in range(dim):
                p = np.zeros(dim)
                for k in range(dim):
                    if k != j:
                        p[k] = tau[i, k]**alpha * (1 / np.abs(ants[i, k] - ants[i, j]))**beta
                p = p / np.sum(p)
                ant[j] = np.random.choice(np.arange(dim), p=p)
            new_ants[i] = ants[i] + np.random.normal(size=dim) * (ub - lb) / 10
        ants = new_ants
        
    return best_sol, best_fit
