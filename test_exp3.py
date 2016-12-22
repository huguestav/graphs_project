from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from ucbmaxn import ucb_max_n


arm_0 = bandits.bernoulliArm(0.5)
arm_1 = bandits.bernoulliArm(0.6)
arm_2 = bandits.bernoulliArm(0.4)

arms = [arm_0, arm_1, arm_2]
n_steps = 300

E = np.array([[1, 0, 0],[0,0,1],[0,0,1]])

alpha = 3.
delta = 2
U = [0,1,2]
gamma = np.min([np.sqrt(1. / (alpha*n_steps)), 0.5])
eta = 2 * gamma


nb_runs = 100
reward = np.zeros(n_steps)
# Run the simulation n_b run times
for t in range(nb_runs):
    reward += exp3G(n_steps, arms, E, U, eta, gamma)[1]

reward = reward / float(nb_runs)
optimal_reward = 0.6 * (np.arange(n_steps)+1)

regret = optimal_reward - np.cumsum(reward)


plt.plot(regret)
plt.show()

