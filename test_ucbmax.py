from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from ucbmaxn import ucb_max_n, ucbn, general_ucbn


arm_0 = bandits.bernoulliArm(0.2)
arm_1 = bandits.bernoulliArm(0.6)
arm_2 = bandits.bernoulliArm(0.4)

arms = [arm_0, arm_1, arm_2]

# E = np.array([[1, 1, 1],[1,1,1],[1,1,1]])
# E = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
E = np.array([[0, 1, 1],[1,0,1],[1,1,0]])


n_steps = 1000
n_runs = 100
loss = np.zeros(n_steps)
for t in range(n_runs):
    # loss += ucbn(n_steps, arms, E)
    # loss += ucb_max_n(n_steps, arms, E)
    loss += general_ucbn(n_steps, arms, E)

loss = loss / float(n_runs)

optimal_loss = 0.6 * (np.arange(n_steps)+1)
regret = optimal_loss - np.cumsum(loss)


plt.plot(regret)
plt.show()



