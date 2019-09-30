import numpy as np


def fill_trans_mat(x, y):
    """

    :param x: int
    :param y: int
    :return: np.array
    """
    trans_mat=np.zeros((3, 4, 4))
    # first action
    trans_mat[0, 1, 1] = 1 - x
    trans_mat[0, 1, 3] = x
    trans_mat[0, 2, 0] = 1 - y
    trans_mat[0, 2, 3] = y
    trans_mat[0, 3, 0] = 1
    # second action
    trans_mat[1, 0, 1] = 1
    # third action
    trans_mat[2, 0, 2] = 1

    return trans_mat


def sum_function(state, utility, trans_mat):
    """

    :param state:
    :param utility:
    :param trans_mat:
    :return:
    """
    A = np.zeros(trans_mat.shape[0])
    for a in range(trans_mat.shape[0]):
            A[a] = np.dot(trans_mat[a, state], utility)
    return A


def rms_error(utils, utils_prime):
    """
    Root mean square error
    :param utils:
    :param utils_prime:
    :return:
    """
    return (1/utils.shape[0])*np.sqrt(np.sum(np.square(utils-utils_prime)))


def value_iteration(rewards, x, y, gamma, epsilon=0.0001):
    """
    Train value_iteration algorithm.
    :param rewards:
    :param x:
    :param y:
    :param gamma:
    :param epsilon:
    :return:
    """
    trans_mat = fill_trans_mat(x,y)
    utils_prime = np.zeros_like(rewards)
    iterate = True
    # find the optimal value function
    while iterate:
        utils = np.copy(utils_prime)
        # update value function at every state
        for i in range(rewards.shape[0]):
            utils_prime[i] = rewards[i]+gamma*np.max(sum_function(i, utils, trans_mat))
        error = rms_error(utils, utils_prime)
        if error<epsilon:
            iterate = False

    # find the policy 
    policy = np.zeros(rewards.shape[0])
    for state in range(rewards.shape[0]):
        policy[state] = np.argmax(sum_function(state, utils, trans_mat))

    return policy, utils


if __name__ =="__main__":
    rewards = np.array([0, 0, 1, 10])
    policy, utils = value_iteration(rewards, x=0.25, y=0.25, gamma=0.9)
    print('utility: ', utils)
    print('policy: ', policy)
