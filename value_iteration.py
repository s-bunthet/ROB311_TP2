import numpy as np

nb_actions = 3
nb_states = 4
x = y =0.25

T = np.zeros(nb_actions, nb_states, nb_states)
# first action
T[0, 1, 1] = 1-x
T[0, 1, 3] = x
T[0, 2, 0] = 1-y
T[0, 2, 3] = y
T[0, 3, 0] = 1

# second action
T[1, 0, 1] = 1

# third action
T[2, 0, 2] = 1

# Reward
R = np.array([0,0,1,10])







def action_value(s, V, T, gamma,R):
    A = np.zeros(nb_actions)
    for a in range(nb_actions):
            A[a] =gamma*np.dot(T[a,s,:],V[])



def value_iteration(epsilon=0.0001, gamma=1):
    Actions = np.zeros(nb_actions)
    Values = np.zeros(nb_states)
    policy = np.zeros([nb_states, nb_actions])
    delta = 0
    # find the optimal value function
    while True:
        # update value function at every state
        for s in range(nb_actions):
            A = action_value(s,Values)
            best_action_value =
        # check the stop condition


    # find the optimal policy

    return policy, V
