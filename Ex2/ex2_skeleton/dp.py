import numpy as np
from lib import GridworldMDP, print_value, print_deterministic_policy, init_value, random_policy

def policy_evaluation_one_step(mdp, V, policy, discount=0.99):
    """ Computes one step of policy evaluation.
    Arguments: MDP, value function, policy, discount factor
    Returns: Value function of policy
    """
    # Init value function array
    V_new = V.copy()

    for s in range(mdp.num_states):
        for a, action_prob in enumerate(policy[s]):
            for prob, next_state, reward, done in mdp.P[s][a]:
                V_new[s] += action_prob * prob * (reward + discount * V[next_state])
    return V_new

def policy_evaluation(mdp, policy, discount=0.99, theta=0.01):
    """ Computes full policy evaluation until convergence.
    Arguments: MDP, policy, discount factor, theta
    Returns: Value function of policy
    """
    # Init value function array
    V = init_value(mdp)
    delta = theta + 1
    while delta >= theta:
        delta = 0
        V_new = policy_evaluation_one_step(mdp, V, policy, discount=discount)
        for s in range(mdp.num_states):
            delta = max(delta, np.abs(V_new[s] - V[s]))
        V = V_new

    return V

def policy_improvement(mdp, V, discount=0.99):
    """Computes greedy policy w.r.t a given MDP and value function.
    Arguments: MDP, value function, discount factor
    Returns: policy
    """
    # Initialize a policy array in which to save the greed policy 
    policy = np.zeros_like(random_policy(mdp))

    for s in range(mdp.num_states):
        q_values = np.zeros(mdp.num_actions)
        for a in range(mdp.num_actions):
            for prob, next_state, reward, done in mdp.P[s][a]:
                q_values[a] += prob * (reward + discount * V[next_state])
        best_action = np.argmax(q_values)
        policy[s] = np.eye(mdp.num_actions)[best_action]

    return policy


def policy_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the policy iteration (PI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """

    # Start from random policy
    policy = random_policy(mdp)


    while True:
        # Evaluate the current policy
        V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
        # Improve the policy
        policy_new = policy_improvement(mdp, V, discount=discount)
        # Check if the policy has changed
        if np.array_equal(policy, policy_new):
            break
        policy = policy_new

    return V, policy

def value_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the value iteration (VI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """
    # Init value function array
    V = init_value(mdp)
    policy = random_policy(mdp)
    while True:
        V = policy_evaluation_one_step(mdp, V, policy, discount=discount)
        policy_new = policy_improvement(mdp, V, discount=discount)
        if np.array_equal(policy, policy_new):
            break
        policy = policy_new
    # Get the greedy policy w.r.t the calculated value function
    policy = policy_improvement(mdp, V)

    return V, policy


if __name__ == "__main__":
    # Create the MDP
    mdp = GridworldMDP([6, 6])
    discount = 0.99
    theta = 0.01

    # Print the gridworld to the terminal
    print('---------')
    print('GridWorld')
    print('---------')
    mdp.render()

    # Create a random policy
    V = init_value(mdp)
    policy = random_policy(mdp)
    # Do one step of policy evaluation and print
    print('----------------------------------------------')
    print('One step of policy evaluation (random policy):')
    print('----------------------------------------------')
    V = policy_evaluation_one_step(mdp, V, policy, discount=discount)
    print_value(V, mdp)

    # Do a full (random) policy evaluation and print
    print('---------------------------------------')
    print('Full policy evaluation (random policy):')
    print('---------------------------------------')
    V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
    print_value(V, mdp)

    # Do one step of policy improvement and print
    # "Policy improvement" basically means "Take greedy action w.r.t given a given value function"
    print('-------------------')
    print('Policy improvement:')
    print('-------------------')
    policy = policy_improvement(mdp, V, discount=discount)
    print_deterministic_policy(policy, mdp)

    # Do a full PI and print
    print('-----------------')
    print('Policy iteration:')
    print('-----------------')
    V, policy = policy_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)

    # Do a full VI and print
    print('---------------')
    print('Value iteration')
    print('---------------')
    V, policy = value_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)