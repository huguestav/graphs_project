import numpy as np
import numpy.matlib

def ucb_max_n(n_steps, arms, E):
    """
    Here we suppose that E does not have any 0 on the diagonal and is symmetric.
    """
    K = len(arms)
    reward = np.zeros(n_steps)

    # Initialize X : total rewards per arm
    # and O : number of draws per arm
    X = np.zeros(K)
    O = np.zeros(K)

    # First draw each arm once
    for k in range(K):
        for i in np.where(E[k,:] != 0)[0]:
            O[i] += 1
            rew = arms[i].sample()
            X[i] += rew
            if i == k:
                reward[k] = rew

    # Do n_steps iterations of the alorithm
    for t in range(n_steps - K):
        Y = X / O + np.sqrt(2.*np.log(t+K) / O)
        i = np.argmax(Y)

        # Select j, the best neighbor of i
        neighbors_i = np.where(E[:,i] != 0)[0]
        j = neighbors_i[np.argmax((X/O)[neighbors_i])]

        # Draw arm j and observe the rewards of the neighbors
        neighbors_j = np.where(E[j,:] != 0)[0]
        for k in neighbors_j:
            O[k] += 1
            rew = arms[k].sample()
            X[k] += rew
            if k == j:
                reward[t + K] = rew

    return reward


def ucbn(n_steps, arms, E):
    """
    Here we suppose that E does not have any 0 on the diagonal and is symmetric.
    """
    K = len(arms)
    reward = np.zeros(n_steps)

    # Initialize X : total rewards per arm
    # and O : number of draws per arm
    X = np.zeros(K)
    O = np.zeros(K)

    # First draw each arm once
    for k in range(K):
        for i in np.where(E[k,:] != 0)[0]:
            O[i] += 1
            rew = arms[i].sample()
            X[i] += rew
            if i == k:
                reward[k] = rew

    # Do n_steps iterations of the alorithm
    for t in range(n_steps - K):
        Y = X / O + np.sqrt(2.*np.log(t+K) / O)
        i = np.argmax(Y)

        neighbors_i = np.where(E[i,:] != 0)[0]

        # Draw arm i and observe the rewards of the neighbors
        for k in neighbors_i:
            O[k] += 1
            rew = arms[k].sample()
            X[k] += rew
            if k == i:
                reward[t + K] = rew

        # Handle the case when there are 0 on the diagonal
        if i not in neighbors_i:
            reward[t + K] = arms[i].sample()

    return reward


def general_ucbn_1(n_steps, arms, E):
    """
    Here we suppose that E does not have any 0 on the diagonal and is symmetric.
    """
    K = len(arms)
    reward = np.zeros(n_steps)

    # Initialize X : total rewards per arm
    # O : number of observations per arm
    # D : number of draws per arm
    X = np.zeros(K)
    O = np.zeros(K)
    D = np.zeros(K)

    # First draw each arm once
    for k in range(K):
        D[k] += 1
        neighbors_k = np.where(E[k,:])[0]
        for i in neighbors_k:
            O[i] += 1
            rew = arms[i].sample()
            X[i] += rew
            if i == k:
                reward[k] = rew

        # Handle the case when there are 0 on the diagonal
        if k not in neighbors_k:
            reward[k] = arms[k].sample()

    # Do n_steps iterations of the alorithm
    for t in range(n_steps - K):
        Y = X / O + np.sqrt(2.*np.log(t+K) / D)
        i = np.argmax(Y)
        D[i] += 1
        neighbors_i = np.where(E[i,:] != 0)[0]

        # Draw arm i and observe the rewards of the neighbors
        for k in neighbors_i:
            O[k] += 1
            rew = arms[k].sample()
            X[k] += rew
            if k == i:
                reward[t + K] = rew

        # Handle the case when there are 0 on the diagonal
        if i not in neighbors_i:
            reward[t + K] = arms[i].sample()

    return reward


def general_ucbn_2(n_steps, arms, E):
    """
    Here we suppose that E does not have any 0 on the diagonal and is symmetric.
    """
    K = len(arms)
    reward = np.zeros(n_steps)

    # Initialize X : total rewards per arm
    # O : number of observations per arm
    # D : number of draws per arm
    X = np.zeros(K)
    O = np.zeros(K)
    D = np.zeros(K)

    # First draw each arm once
    for k in range(K):
        D[k] += 1
        neighbors_k = np.where(E[k,:])[0]
        for i in neighbors_k:
            O[i] += 1
            rew = arms[i].sample()
            X[i] += rew
            if i == k:
                reward[k] = rew

        # Handle the case when there are 0 on the diagonal
        if k not in neighbors_k:
            reward[k] = arms[k].sample()

    # Do n_steps iterations of the alorithm
    for t in range(n_steps - K):
        D_val = np.zeros(K)
        for k in range(K):
            non_zero_idx = np.where(E[k,:])[0]
            if len(non_zero_idx):
                # If k observes rewards, take the min number of rewards among
                # the neighbors of k
                D_val[k] = np.min(O[np.where(E[k,:])[0]])
            else:
                # Take the number of draws of k
                D_val[k] = max(D[k], O[k])
                # D_val[k] = D[k] + O[k]
                # D_val[k] = O[k]
                # D_val[k] = D[k]
                # print O[k] == D[k]

                # print "O:", O
                # print "D:", D
                # D_val[k] = float(D[k])

        Y = X / O + np.sqrt(2.*np.log(t+K) / D_val)
        i = np.argmax(Y)
        neighbors_i = np.where(E[i,:] != 0)[0]


        # print "Y:", Y
        # print "i:", i
        # print ""

        # Draw arm i and observe the rewards of the neighbors
        D[i] += 1
        for k in neighbors_i:
            O[k] += 1
            rew = arms[k].sample()
            X[k] += rew
            if k == i:
                reward[t + K] = rew

        # Handle the case when there are 0 on the diagonal
        if i not in neighbors_i:
            reward[t + K] = arms[i].sample()

    return reward


def general_ucbn_3(n_steps, arms, E):
    """
    Here we suppose that E does not have any 0 on the diagonal and is symmetric.
    """
    K = len(arms)
    reward = np.zeros(n_steps)

    # Initialize X : total rewards per arm
    # O : number of observations per arm
    # D : number of draws per arm
    X = np.zeros(K)
    O = np.zeros(K)
    D = np.zeros(K)

    # First draw each arm once
    for k in range(K):
        D[k] += 1
        neighbors_k = np.where(E[k,:])[0]
        for i in neighbors_k:
            O[i] += 1
            rew = arms[i].sample()
            X[i] += rew
            if i == k:
                reward[k] = rew

        # Handle the case when there are 0 on the diagonal
        if k not in neighbors_k:
            reward[k] = arms[k].sample()

    # Do n_steps iterations of the alorithm
    for t in range(n_steps - K):
        D_arg = np.zeros(K, dtype=np.int)
        D_val = np.zeros(K, dtype=np.int)
        for k in range(K):
            non_zero_idx = np.where(E[k,:])[0]

            if len(non_zero_idx):
                # If k observes rewards, take the min number of rewards among
                # the neighbors of k
                min_non_zero_idx = np.argmin(O[non_zero_idx])
                D_arg[k] = non_zero_idx[min_non_zero_idx]
                D_val[k] = O[D_arg[k]]
            else:
                # Take the number of draws of k
                D_arg[k] = float(k)
                D_val[k] = max(D[k], O[k])
                # D_val[k] = D[k]

        Y = X / O + np.sqrt(2.*np.log(t+K) / D_val)
        i = np.argmax(Y)

        # Neighbor of i with smallest number of observations (i -> k)
        k = D_arg[i]
        neighbors_k = np.where(E[:,k] != 0)[0]
        if len(neighbors_k):
            # Consider as a candidate : j which is the neighbor of k
            # with the highest expected value.
            j = neighbors_k[np.argmax((X/O)[neighbors_k])]

            # Select the best arm between i and j
            Y = X / O
            if Y[j] > Y[i]:
                i = j

        neighbors_i = np.where(E[i,:] != 0)[0]

        # Draw arm i and observe the rewards of the neighbors
        D[i] += 1
        for k in neighbors_i:
            O[k] += 1
            rew = arms[k].sample()
            X[k] += rew
            if k == i:
                reward[t + K] = rew

        # Handle the case when there are 0 on the diagonal
        if i not in neighbors_i:
            reward[t + K] = arms[i].sample()

    return reward


def ucb1(n_steps, arms):
    """
    This is the basic UCB1 algorithm.
    Here we suppose that E does not have any 0 on the diagonal and is symmetric.
    """
    K = len(arms)
    reward = np.zeros(n_steps)

    # Initialize X : total rewards per arm and O : number of draws per arm
    X = np.zeros(K)
    O = np.zeros(K)

    # First draw each arm once
    for k in range(K):
        O[k] += 1.
        rew = arms[k].sample()
        X[k] += rew
        reward[k] =  rew

    # Do n_steps iterations of the alorithm
    for t in range(n_steps - K):
        Y = X / O + np.sqrt(2.*np.log(t+K) / O)
        i = np.argmax(Y)
        rew = arms[i].sample()
        O[i] += 1.
        X[i] += rew
        reward[t + K] = rew
    return reward
