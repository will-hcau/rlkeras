import numpy as np

class GreedyQPolicy():

    """Implementation of Greedy Q Policy

`   Always select the action with the largest Q value

    """

    def __init__(self):
        return

    def select_action(self, q_values):
        action = np.argmax(q_values)
        return action

class EpsGreedyQPolicy():

    """Implementation of Epsilon Greedy Q Policy

`   To ensure the agent will explore the environment with a
    certein probablity. Define a value "epsilon" so that
    the agent will random sample action at a controled probabiliy.

    """

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

    """Implementation of Boltzmann Q Policy

    aks Boltzmann exploration in the paper

`   The Boltzmann exploration policy is intended for discrete action spaces.
    It assumes that each of the possible actions has some value assigned to it
    (such as the Q value), and uses a softmax function to convert these values
    into a distribution over the actions. It then samples the action for playing
    out of the calculated distribution. An additional temperature schedule can be
    given by the user, and will control the steepness of the softmax function.

    """

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