from matplotlib import pyplot as plt
import numpy as np
import bandits
from exp3G import exp3G
from ucbmaxn import ucb_max_n
from sklearn import linear_model


arm_0 = bandits.bernoulliArm(0.2)
arm_1 = bandits.bernoulliArm(0.6)
arm_2 = bandits.bernoulliArm(0.4)


arms_a = [arm_0, arm_1, arm_2]

n_steps = 3000

means_a = [arm_0.mean, arm_1.mean, arm_2.mean]


#E = np.array([[0, 1, 0],[1,0,1],[1,1,0]]) #weakly observable
E = np.array([[1, 0, 0],[0,1,0],[0,0,1]]) #strongly observable
#E = np.array([[0,1,0],[0,1,1],[0,1,1]]) #not observable

alpha = 1.
delta = 2.
U = [0,1,2]

gamma = np.min([np.sqrt(1. / (alpha*n_steps)), 0.5])

eta = 2 * gamma

nb_runs = 200
loss_a = np.zeros(n_steps)

# Run the simulation n_b run times
for t in range(nb_runs):
    loss_a += exp3G(n_steps, arms_a, E, U, eta, gamma)[1]

loss_a = loss_a / float(nb_runs)


optimal_loss_a = min(means_a) * (np.arange(n_steps)+1)

regret_a = np.cumsum(loss_a) - optimal_loss_a

plt.plot(regret_a, label="1")
plt.xlabel("Number of iterations")
plt.ylabel("regret")
plt.show()

tu = range(1,n_steps)

on = np.ones((n_steps-1,1))


x_temp = np.log(tu).reshape((n_steps-1,1))
y = np.log(regret_a[tu]).reshape((n_steps-1,1))


x = np.concatenate((x_temp,on), axis=1)

W = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

r = y - x.dot(W)

Var = (1/(n_steps - 1)) * r.T.dot(r)

print(Var)
print(W)






