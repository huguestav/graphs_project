import numpy as np


def ts(n_steps, arms, E):
    """
    Here we suppose that E does not have any 0 on the diagonal and is symmetric.
    """
    K = len(arms)
    reward = np.zeros(n_steps)

    # Initialize X : total rewards per arm
    # and O : number of draws per arm
    X = np.zeros(K)
    O = np.zeros(K)

    # Do n_steps iterations of the algorithm
    for t in range(n_steps):
        theta = np.random.beta(X+1,O-X+1)

        # Choose the arm that maximizes theta
        i = np.argmax(theta)
        # print i
        neighbors_i = np.where(E[i,:] != 0)[0]

        # Draw arm i and observe the rewards of the neighbors
        for k in neighbors_i:
            O[k] += 1
            rew = arms[k].sample()
            X[k] += rew
            if k == i:
                reward[t] = rew

    return reward



def ts_max(n_steps, arms, E):
    """
    Here the feedback graph E is general
    """
    K = len(arms)
    reward = np.zeros(n_steps)

    # Initialize X : total rewards per arm
    # and O : number of draws per arm
    X = np.zeros(K)
    O = np.zeros(K)
    D = np.zeros(K)

    # Do n_steps iterations of the algorithm
    for t in range(n_steps):
        theta = np.random.beta(X+1,O-X+1)

        # Choose the arm that maximizes theta
        i = np.argmax(theta)

        # Consider j, the best neighbor of i
        Y = X/O
        neighbors_i = np.where(E[:,i] != 0)[0]
        j = neighbors_i[np.argmax(Y[neighbors_i])]
        if Y[j] > Y[i]:
            i = j

        # Draw arm j and observe the rewards of the neighbors
        neighbors_i = np.where(E[i,:] != 0)[0]
        D[i] += 1
        for k in neighbors_i:
            O[k] += 1
            rew = arms[k].sample()
            X[k] += rew
            if k == i:
                reward[t] = rew

        # Handle the case when there are 0 on the diagonal
        if i not in neighbors_i:
            reward[t] = arms[i].sample()

    return reward


def general_ts(n_steps, arms, E):
    """
    """
    K = len(arms)
    reward = np.zeros(n_steps)

    # Initialize X : total rewards per arm
    # O : number of observations per arm
    X = np.zeros(K)
    O = np.zeros(K)

    # Do n_steps iterations of the algorithm
    for t in range(n_steps):
        theta = np.random.beta(X+1,O-X+1)

        # Choose the arm that maximizes theta
        i = np.argmax(theta)
        neighbors_i = np.where(E[i,:] != 0)[0]

        # Draw arm i and observe the rewards of the neighbors
        for k in neighbors_i:
            O[k] += 1
            rew = arms[k].sample()
            X[k] += rew
            if k == i:
                reward[t] = rew

        # Handle the case when there are 0 on the diagonal
        if i not in neighbors_i:
            reward[t] = arms[i].sample()

    return reward
