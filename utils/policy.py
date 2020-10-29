import numpy as np

class GreedyQPolicy():

    def __init__(self):
        return

    def select_action(self, q_values):
        action = np.argmax(q_values)
        return action

class EpsGreedyQPolicy():

    def __init__(self, eps=.1):
        self.eps = eps

    def select_action(self, q_values):

        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)

        return action

class BoltzmannQPolicy():

    def __init__(self, tau=1., clip=(-500., 500.)):
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        q_values = q_values.astype('float64')
        nb_actions = q_values.size

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs[0])

        return action