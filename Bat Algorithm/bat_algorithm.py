import numpy as np
import math

class BatAlgorithm:
    def __init__(self, objective_func, dim, population_size=100, max_iter=1000, A=1, alpha=0.9, gamma=0.9, fmin=-5.12, fmax=5.12):
        self.objective_func = objective_func
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.fmin = fmin
        self.fmax = fmax
        self.population = np.zeros((population_size, dim))
        self.velocities = np.zeros((population_size, dim))
        self.fitness = np.zeros(population_size)
        self.loudness = np.ones(population_size)
        self.pulse_rate = np.zeros(population_size)
        self.location = np.zeros((population_size, dim))
        self.fitness_best = np.inf
        self.location_best = np.zeros(dim)
        
    def init_population(self):
        self.population = np.random.uniform(self.fmin, self.fmax, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)
        self.loudness = np.ones(self.population_size)
        self.pulse_rate = np.zeros(self.population_size)
        self.location = self.population.copy()
        
    def evaluate_fitness(self):
        for i in range(self.population_size):
            self.fitness[i] = self.objective_func(self.population[i])
            if self.fitness[i] < self.fitness_best:
                self.fitness_best = self.fitness[i]
                self.location_best = self.population[i].copy()
                
    def move_bats(self, iter):
        for i in range(self.population_size):
            self.velocities[i] += (self.location_best - self.population[i]) * self.A * np.exp(-self.alpha * iter)
            self.population[i] += self.velocities[i]
            if np.random.uniform(0, 1) > self.pulse_rate[i]:
                self.population[i] = self.location_best + self.gamma * np.random.uniform(-1, 1, self.dim) * (self.fmax - self.fmin)
            self.population[i] = np.clip(self.population[i], self.fmin, self.fmax)
            
    def update_bats(self, iter):
        for i in range(self.population_size):
            if np.random.uniform(0, 1) < self.loudness[i]:
                self.pulse_rate[i] = np.exp(-self.gamma * iter)
                self.loudness[i] *= self.alpha
                self.location[i] = self.population[i].copy()
    
    def optimize(self):
        self.init_population()
        for i in range(self.max_iter):
            self.evaluate_fitness()
            print(f"Iteration {i+1}: Best fitness: {self.fitness_best}")
            self.move_bats(i)
            self.update_bats(i)
        return self.fitness_best, self.location_best
