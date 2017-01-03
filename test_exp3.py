from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from ucbmaxn import ucb_max_n

# arm_0 = bandits.bernoulliArm(0.2)
# arm_1 = bandits.bernoulliArm(0.6)
# arm_2 = bandits.bernoulliArm(0.4)

#arm_0 = bandits.gaussianArm(0.5, 1)
#arm_1 = bandits.gaussianArm(0, 0.6)
#arm_2 = bandits.gaussianArm(1, 0.4)

#arm_0 = bandits.expArm(1.0)
#arm_1 = bandits.expArm(1.6)
#arm_2 = bandits.expArm(1.4)

arm_0 = bandits.betaArm(0.5, 0.5)
arm_1 = bandits.betaArm(5,1)
arm_2 = bandits.betaArm(1,3)

arms = [arm_0, arm_1, arm_2]
n_steps = 3000

# E = np.array([[0, 1, 0],[0,0,1],[1,0,0]])
E = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
# E = np.array([[1, 1, 1],[1,1,1],[1,1,1]])
means = [arm_0.mean, arm_1.mean, arm_2.mean]

n_steps = 300

#E = np.array([[0, 1, 0],[1,0,1],[1,1,0]]) #weakly observable
E = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #strongly observable
# E = np.array([[0,1,0],[0,1,1],[0,1,1]]) #not observable

#E = np.array([[1, 0, 0],[0,0,1],[0,0,1]])

alpha = 1.
delta = 2
U = [0,1,2]
gamma = np.min([np.sqrt(1. / (alpha*n_steps)), 0.5])
eta = 2 * gamma

nb_runs = 100
loss = np.zeros(n_steps)
reward = np.zeros(n_steps)

# Run the simulation n_b run times
for t in range(nb_runs):
    loss += exp3G(n_steps, arms, E, U, eta, gamma)[1]

loss = loss / float(nb_runs)

optimal_loss = min(means) * (np.arange(n_steps)+1)
regret = np.cumsum(loss) - optimal_loss



plt.plot(regret)
plt.show()

