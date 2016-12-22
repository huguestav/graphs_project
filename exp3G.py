import numpy as np
import random


def draw_finite(p):
    """
    Draw a sample from a distribution on a finite set
    that takes value k with probability p(k)
    """
    q = np.cumsum(p)
    u = random.random()
    i = 0
    while u > q[i]:
        i += 1
    return i

def exp3G(n_steps, arms, E, U, eta, gamma):
    """
    - n_steps is the number of steps to run the algorithm for
    - E is the feedback graph : E(i,j)=1 when j is in the out-neighborhood of i
    - U is the exploration set
    - eta > 0 is the learning rate
    - gamma (in [0,1]) is the exploration rate
    """
    K = len(arms)

    # u is the uniform distribution over U
    u = 1. / len(U) * np.in1d(range(K), U)

    # q is initialized as the uniform distribution over V
    q = 1. / K * np.ones(K)

    reward = np.zeros(n_steps)
    draws = np.zeros(n_steps)
    for t in range(n_steps):
        # Update p
        p = (1-gamma) * q + gamma * u

        # Draw acion from the possible actions with distribution p
        action = draw_finite(p)
        draws[t] = action

        rew = np.zeros(K)
        for k in range(K):
            side_rew = arms[k].sample()

            if k == action:
                reward[t] += side_rew

            # If k is in the out-neighborhood of action
            if E[action,k]:
                rew[k] = side_rew / p.T.dot(E[:,k])

        # Update q
        temp = q * np.exp(-eta * rew)
        q = temp / np.sum(temp)

    return draws,reward




