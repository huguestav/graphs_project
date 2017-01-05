from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from ucbmaxn import ucb_max_n

arm_0 = bandits.bernoulliArm(0.2)
arm_1 = bandits.bernoulliArm(0.6)
arm_2 = bandits.bernoulliArm(0.4)

#arm_0 = bandits.gaussianArm(0.5, 1)
#arm_1 = bandits.gaussianArm(0, 0.6)
#arm_2 = bandits.gaussianArm(1, 0.4)

#arm_0 = bandits.expArm(1.0)
#arm_1 = bandits.expArm(1.6)
#arm_2 = bandits.expArm(1.4)

# arm_0 = bandits.betaArm(0.5, 0.5)
# arm_1 = bandits.betaArm(5,1)
# arm_2 = bandits.betaArm(1,3)


def run_simu(arms, E, n_steps, n_runs, U, eta):
    means = [arm.mean for arm in arms]
    loss = np.zeros(n_steps)

    # Run the simulation n_b run times
    for t in range(n_runs):
        loss += exp3G(n_steps, arms, E, U, eta, gamma)[1]
    loss = loss / float(n_runs)

    optimal_loss = min(means) * (np.arange(n_steps)+1)
    regret = np.cumsum(loss) - optimal_loss

    return regret



arms = [arm_0, arm_1, arm_2]
means = [arm_0.mean, arm_1.mean, arm_2.mean]

E = np.array([[0, 1, 0],[1,0,1],[1,1,0]]) #weakly observable
# E = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #strongly observable
# E = np.array([[0,1,0],[0,1,1],[0,1,1]]) #not observable


n_steps = 1000
n_runs = 100

# First bandit
E = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
alpha = 1.
delta = 3
U = [0,1,2]
gamma = np.min([np.sqrt(1. / (alpha*n_steps)), 0.5])
eta = 2 * gamma

regret_1 = run_simu(arms, E, n_steps, n_runs, U, eta)


# # second bandit
# E = np.array([[0, 1, 0],[0,0,1],[1,0,0]])
# alpha = 3.
# delta = 3.
# K = 3
# U = [0,1,2]
# gamma = np.min([np.power(delta*np.log(K) / float(n_steps),1./3), 0.5])
# eta = gamma**2 / delta

# regret_2 = run_simu(arms, E, n_steps, n_runs, U, eta)


# # Third bandit
# E = np.array([[0, 0, 0],[0,0,0],[1,0,0]])
# alpha = 3.
# delta = 3.
# K = 3
# U = [0,1,2]
# gamma = np.min([np.power(delta*np.log(K) / float(n_steps),1./3), 0.5])
# eta = gamma**2 / delta

# regret_3 = run_simu(arms, E, n_steps, n_runs, U, eta)






first, = plt.plot(regret_2)
plt.show()


# abscisse = np.arange(n_steps) + 1

# first, = plt.plot(abscisse, regret_1, label="Strongly observable")
# second, = plt.plot(abscisse, regret_2, label="Weakly observable")
# third, = plt.plot(abscisse, regret_3, label="Not observable")
# plt.legend(handles=[first, second, third], loc=0)
# plt.show()

