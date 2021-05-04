import numpy as np
from collections import defaultdict


class AgentWithEpsilon:

    def __init__(self, env, num_episodes, epsilon=0.1, alpha=0.1, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.env = env
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.nA = env.action_space.n
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.random() < self.epsilon:  # Explore. Sample uniformly from all actions
            action = self.env.action_space.sample()
        else:  # Exploit. Take the greedy action
            action = np.argmax(self.Q[state])

        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        old_Q = self.Q[state][action]
        action_probabs = self._epsilon_greedy_probabilities(next_state)
        expected_value = np.dot(self.Q[next_state], action_probabs)
        self.Q[state][action] = old_Q + self.alpha * (reward + self.gamma * expected_value - old_Q)

    def _epsilon_greedy_probabilities(self, state):
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        policy_s[best_a] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy_s

    def update_epsilon(self, i_episode):
        pass
