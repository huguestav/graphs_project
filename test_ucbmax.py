from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from ucbmaxn import ucb_max_n, ucbn

#arm_0 = bandits.bernoulliArm(0.2)
#arm_1 = bandits.bernoulliArm(0.6)
#arm_2 = bandits.bernoulliArm(0.4)

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
means = [arm_0.mean, arm_1.mean, arm_2.mean]

#E = np.array([[1, 1, 1],[1,1,1],[1,1,1]]) #weakly observable
E = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #strongly observable
#E = np.array([[0, 1, 0],[1,0,1],[1,1,0]]) #weakly observable

n_steps = 100
n_runs = 100
loss = np.zeros(n_steps)
for t in range(n_runs):
    # loss += ucbn(n_steps, arms, E)
    loss += ucb_max_n(n_steps, arms, E)
loss = loss / float(n_runs)

optimal_loss = max(means) * (np.arange(n_steps)+1)
regret = optimal_loss - np.cumsum(loss)


plt.plot(regret)
plt.show()






