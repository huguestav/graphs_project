from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from thompson_sampling import ts, ts_max, general_ts
from sklearn import linear_model
import matplotlib.pyplot as plt

arm_0 = bandits.bernoulliArm(0.2)
arm_1 = bandits.bernoulliArm(0.6)
arm_2 = bandits.bernoulliArm(0.4)

#arm_0 = bandits.gaussianArm(0.5, 1)
#arm_1 = bandits.gaussianArm(0, 0.6)
#arm_2 = bandits.gaussianArm(1, 0.4)

#arm_0 = bandits.expArm(1.0)
#arm_1 = bandits.expArm(1.6)
#arm_2 = bandits.expArm(1.4)

arm_3 = bandits.betaArm(0.5, 0.5)
arm_4 = bandits.betaArm(5,1)
arm_5 = bandits.betaArm(1,3)

arms_a = [arm_0, arm_1, arm_2]
arms_b = [arm_3, arm_4, arm_5]

means_a = [arm_0.mean, arm_1.mean, arm_2.mean]
means_b = [arm_3.mean, arm_4.mean, arm_5.mean]

#E = np.array([[1, 1, 1],[1,1,1],[1,1,1]]) #weakly observable
E = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #strongly observable
#E = np.array([[0, 1, 0],[1,0,1],[1,1,0]]) #weakly observable
#

n_steps = 300
n_runs = 100

loss_a = np.zeros(n_steps)
loss_b = np.zeros(n_steps)

for t in range(n_runs):
     #loss += ts(n_steps, arms, E)
    # loss += ts_max(n_steps, arms, E)
    loss_a += general_ts(n_steps, arms_a, E)
    loss_b += general_ts(n_steps, arms_b, E)

loss_a = loss_a / float(n_runs)
loss_b = loss_b / float(n_runs)

optimal_loss_a = max(means_a) * (np.arange(n_steps)+1)
optimal_loss_b = max(means_b) * (np.arange(n_steps)+1)

regret_a = optimal_loss_a - np.cumsum(loss_a)
regret_b = optimal_loss_b - np.cumsum(loss_b)

plt.plot(regret_a)
plt.plot(regret_b)
plt.show()

tu = range(1,n_steps)

x = np.log(tu).reshape((299,1))
y = np.log(regret_a[tu]).reshape((299,1))

regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Coefficients: \n', regr.coef_)





