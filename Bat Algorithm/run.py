import numpy as np
import math


n = 10 # number of bats
d = 2 # number of dimensions in the search space
max_iter = 1000 # maximum number of iterations
ub = 5 # upper bound of the search space
lb = -5 # lower bound of the search space
f_min = 0 # minimum frequency
f_max = 2 # maximum frequency
alpha = 0.9 # constant for updating the location of bats
gamma = 0.9 # constant for updating the loudness of bats
f = 1e-6 # convergence criterion
A = np.ones(n) # loudness of bats
f_opt = 0 # optimal value



# define the objective function to be optimized
def obj_func(x):
    return x ** 2

# define the Bat Algorithm function
def bat_algorithm(n, d, max_iter, ub, lb, f_min, f_max, alpha, gamma, f, A, f_opt):
    # initialize bats
    bats = np.random.uniform(lb, ub, (n, d))
    v = np.zeros((n, d))
    f_vals = np.zeros(n)
    for i in range(n):
        f_vals[i] = obj_func(bats[i])

    # find the best bat
    best_bat_idx = np.argmin(f_vals)
    best_bat = bats[best_bat_idx]
    best_f_val = f_vals[best_bat_idx]

    # print the initial best fitness value and corresponding position
    print(f"Iteration 0: Best fitness value = {best_f_val}, Best position = {best_bat}")

    # iterate until convergence or maximum iterations
    for t in range(1, max_iter+1):
        for i in range(n):
            # update frequency and velocity
            f_i = f_min + (f_max - f_min) * np.random.rand()
            v[i] += (bats[i] - best_bat) * f_i
            bats[i] += v[i]

            # check boundaries
            for j in range(d):
                if bats[i][j] > ub:
                    bats[i][j] = ub
                    v[i][j] *= -1
                elif bats[i][j] < lb:
                    bats[i][j] = lb
                    v[i][j] *= -1

            # check if the bat emits a pulse
            if np.random.rand() > A[i]:
                continue

            # update frequency, location, and loudness
            f_i = f_min + (f_max - f_min) * np.random.rand()
            bats[i] = best_bat + alpha * np.random.normal(0, 1, d)
            bats[i] += (f_i * np.random.rand(d) - 0.5) * gamma
            f_vals[i] = obj_func(bats[i])
            A[i] *= 0.9

            # update the best bat
            if f_vals[i] < best_f_val:
                best_bat = bats[i]
                best_f_val = f_vals[i]

        # update the maximum frequency and loudness
        f_max *= 0.95
        A_min = 0.01
        A = np.maximum(A, A_min)

        # print the best fitness value and corresponding position at each iteration
        print(f"Iteration {t}: Best fitness value = {best_f_val}, Best position = {best_bat}")

        # check if optimal value has been reached
        if abs(best_f_val - f_opt) < f:
            break

    return best_bat, best_f_val
