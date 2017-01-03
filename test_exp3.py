from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from ucbmaxn import ucb_max_n


arm_0 = bandits.bernoulliArm(0.5)
arm_1 = bandits.bernoulliArm(0.6)
arm_2 = bandits.bernoulliArm(0.4)

arms = [arm_0, arm_1, arm_2]
n_steps = 3000

# E = np.array([[0, 1, 0],[0,0,1],[1,0,0]])
E = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
# E = np.array([[1, 1, 1],[1,1,1],[1,1,1]])

alpha = 1.
delta = 2
U = [0,1,2]
gamma = np.min([np.sqrt(1. / (alpha*n_steps)), 0.5])
eta = 2 * gamma


nb_runs = 100
loss = np.zeros(n_steps)
# Run the simulation n_b run times
for t in range(nb_runs):
    loss += exp3G(n_steps, arms, E, U, eta, gamma)[1]

loss = loss / float(nb_runs)
# optimal_loss = 0.6 * (np.arange(n_steps)+1)
# regret = optimal_loss - np.cumsum(loss)

optimal_loss = 0.4 * (np.arange(n_steps)+1)
regret = np.cumsum(loss) - optimal_loss




plt.plot(regret)
plt.show()

