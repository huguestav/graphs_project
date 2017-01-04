from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from ucbmaxn import ucb_max_n
from sklearn import linear_model


arm_0 = bandits.bernoulliArm(0.2)
arm_1 = bandits.bernoulliArm(0.6)
arm_2 = bandits.bernoulliArm(0.4)

#arm_3 = bandits.bernoulliArm(0.9)
#arm_4 = bandits.bernoulliArm(0.89)
#arm_5 = bandits.bernoulliArm(0.92)

#arm_3 = bandits.gaussianArm(0, 0.2)
#arm_4 = bandits.gaussianArm(0, 0.6)
#arm_5 = bandits.gaussianArm(0, 0.4)

#arm_3 = bandits.expArm(1.0)
#arm_4 = bandits.expArm(1.6)
#arm_5 = bandits.expArm(1.4)

arm_3 = bandits.betaArm(0.5, 3)
arm_4 = bandits.betaArm(5,1)
arm_5 = bandits.betaArm(1,3)

arms_a = [arm_0, arm_1, arm_2]
arms_b = [arm_3, arm_4, arm_5]
n_steps = 300

# E = np.array([[0, 1, 0],[0,0,1],[1,0,0]])
#E = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
# E = np.array([[1, 1, 1],[1,1,1],[1,1,1]])
means_a = [arm_0.mean, arm_1.mean, arm_2.mean]
means_b = [arm_3.mean, arm_4.mean, arm_5.mean]

#n_steps = 100

#E = np.array([[0, 1, 0],[1,0,1],[1,1,0]]) #weakly observable
E = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #strongly observable
#E = np.array([[0,1,0],[0,1,1],[0,1,1]]) #not observable

#E = np.array([[0, 1, 1],[1,0,1],[1,1,0]]) #strongly observable

#E = np.array([[1, 0, 0],[0,0,1],[0,0,1]])

alpha = 1.
delta = 2.
#U = [0,1,2]
U = [0,1,2]

#gamma = np.min(((delta*np.log(len(arms)))/n_steps)**(1/3), 0.5)
gamma = np.min([np.sqrt(1. / (alpha*n_steps)), 0.5])

eta = 2 * gamma
#eta = gamma**2/6

nb_runs = 100
loss_a = np.zeros(n_steps)
loss_b = np.zeros(n_steps)

# Run the simulation n_b run times
for t in range(nb_runs):
    loss_a += exp3G(n_steps, arms_a, E, U, eta, gamma)[1]
    loss_b += exp3G(n_steps, arms_b, E, U, eta, gamma)[1]

loss_a = loss_a / float(nb_runs)
loss_b = loss_b / float(nb_runs)


optimal_loss_a = min(means_a) * (np.arange(n_steps)+1)
optimal_loss_b = min(means_b) * (np.arange(n_steps)+1)

regret_a = np.cumsum(loss_a) - optimal_loss_a
regret_b = np.cumsum(loss_b) - optimal_loss_b

#log_regret = np.log(regret)
#regV = range(1,n_steps)
#log_regV = np.log(regV)
#
#regVT = regret[regV]/log_regV
##/log_regV
##/log_regV
##/log_regV
##/log_regV
#
#log_v = np.log(regVT)

plt.plot(regret_a, label="1")
plt.plot(regret_b, label="2")
plt.xlabel("Number of iterations")
plt.ylabel("regret")
plt.show()

#log_regV.reshape((299,1))
#log_v.reshape((299,1))
#
#regr = linear_model.LinearRegression()
#regr.fit(log_regV, log_v)
#
#print('slope', regr.coeff )





