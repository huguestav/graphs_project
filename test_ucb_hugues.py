from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from ucbmaxn import *


def run_simu(arms, E, n_steps, n_runs, algo=ucb_max_n):
    means = [arm.mean for arm in arms]
    reward = np.zeros(n_steps)

    # Run the simulation n_runs times
    for t in range(n_runs):
        reward += algo(n_steps, arms, E)
    reward = reward / float(n_runs)

    optimal_reward = max(means) * (np.arange(n_steps)+1)
    regret = optimal_reward - np.cumsum(reward)

    return regret



arm_0 = bandits.bernoulliArm(0.5)
arm_1 = bandits.bernoulliArm(0.6)
arm_2 = bandits.bernoulliArm(0.57)

#arm_0 = bandits.gaussianArm(0.5, 1)
#arm_1 = bandits.gaussianArm(0, 0.6)
#arm_2 = bandits.gaussianArm(1, 0.4)

#arm_0 = bandits.expArm(1.0)
#arm_1 = bandits.expArm(1.6)
#arm_2 = bandits.expArm(1.4)

# arm_0 = bandits.betaArm(0.5, 0.5)
# arm_1 = bandits.betaArm(5,1)
# arm_2 = bandits.betaArm(1,3)

arms = [arm_0, arm_1, arm_2]

# E = np.array([[1, 1, 1],[1,1,1],[1,1,1]]) #weakly observable
#E = np.array([[0, 1, 0],[1,0,1],[1,1,0]]) #weakly observable

n_steps = 200
n_runs = 1000



# E = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
E = np.array([[1, 1, 0],[0,1,0],[1,1,1]])
regret_1 = run_simu(arms, E, n_steps, n_runs, algo=general_ucbn_3)

# E = np.array([[0, 1, 0],[0,0,1],[1,0,0]])
regret_2 = run_simu(arms, E, n_steps, n_runs, algo=general_ucbn_2)


# plt.plot(regret_1)
# plt.show()




abscisse = np.arange(n_steps) + 1

first, = plt.plot(abscisse, regret_1, label="1")
second, = plt.plot(abscisse, regret_2, label="2")
plt.legend(handles=[first, second], loc=0)
plt.show()



