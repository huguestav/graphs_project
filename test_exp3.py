from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from ucbmaxn import ucb_max_n
from sklearn import linear_model


arm_0 = bandits.bernoulliArm(0.2)
arm_1 = bandits.bernoulliArm(0.6)
arm_2 = bandits.bernoulliArm(0.4)

#arm_3 = bandits.bernoulliArm(0.01)
#arm_4 = bandits.bernoulliArm(0.02)
#arm_5 = bandits.bernoulliArm(0.03)


#arm_3 = bandits.gaussianArm(0, 0.2)
#arm_4 = bandits.gaussianArm(0, 0.6)
#arm_5 = bandits.gaussianArm(0, 0.4)

#arm_3 = bandits.expArm(4.09)
#arm_4 = bandits.expArm(0.91)
#arm_5 = bandits.expArm(1.68)

#arm_3 = bandits.betaArm(0.5, 3)
#arm_4 = bandits.betaArm(5,1)
#arm_5 = bandits.betaArm(1,3)

arms_a = [arm_0, arm_1, arm_2]
#arms_b = [arm_3, arm_4, arm_5]

means_a = [arm_0.mean, arm_1.mean, arm_2.mean]
#means_b = [arm_3.mean, arm_4.mean, arm_5.mean]


#E = np.array([[0, 1, 0],[1,0,1],[1,1,0]]) #weakly observable
E = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #strongly observable
#E = np.array([[0,1,0],[0,1,1],[0,1,1]]) #not observable
#E = np.array([[0, 1, 1],[1,0,1],[1,1,0]]) #strongly observable


n_steps = 3000
E = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
alpha = 1.
delta = 2.
U = [0,1,2]

# Strong case
gamma = np.min([np.sqrt(1. / (alpha*n_steps)), 0.5])
eta = 2 * gamma

# Weak case
#gamma = np.min(((delta*np.log(len(arms)))/n_steps)**(1/3), 0.5)
#eta = gamma**2/float(delta)


nb_runs = 200
loss_a = np.zeros(n_steps)
#loss_b = np.zeros(n_steps)

# Run the simulation n_b run times
for t in range(nb_runs):
    loss_a += exp3G(n_steps, arms_a, E, U, eta, gamma)[1]
#    loss_b += exp3G(n_steps, arms_b, E, U, eta, gamma)[1]

loss_a = loss_a / float(nb_runs)
#loss_b = loss_b / float(nb_runs)


optimal_loss_a = min(means_a) * (np.arange(n_steps)+1)
#optimal_loss_b = min(means_b) * (np.arange(n_steps)+1)

regret_a = np.cumsum(loss_a) - optimal_loss_a
#regret_b = np.cumsum(loss_b) - optimal_loss_b

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
#plt.plot(regret_b, label="2")
plt.xlabel("Number of iterations")
plt.ylabel("regret")
plt.show()

tu = range(1,n_steps)
on = np.ones((n_steps-1,1))


x = np.log(tu).reshape((n_steps-1,1))
y = np.log(regret_a[tu]).reshape((n_steps-1,1))
x = np.concatenate((x,on), axis=1)

# Do a linear regression
W = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
r = y - x.dot(W)

# Compute the variance
variance = (1./(n_steps - 1)) * r.T.dot(r)
variance = variance[0,0]

print("variance : {variance}".format(variance=variance))
print("slope : {slope}".format(slope=W[0,0]))
print("value_at_zero : {value_at_zero}".format(value_at_zero=W[1,0]))



#regr = linear_model.LinearRegression()
#regr.fit(x, y)

#print('Coefficients: \n', regr.coef_)
#print('Variance score: %.2f', regr.score(x, y))

