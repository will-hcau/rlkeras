from collections import deque
import numpy as np
import random


class RandomReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        """ Store transition into replay buffer "D"

		Refering to the DQN paper (S, A, R, S t+1, terminate)
		should be stored into a buffer with limited size.
		When hitting the maximum size of buffer, the oldest
		transition will be discard.

        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, num_of_step=1):
        """  Sampling

        Random sample a minibatch from the replay buffer

        """
        sample_data = []

        sample_indices = np.random.random_integers(0, len(self.buffer) - num_of_step, size=batch_size)

        for s in sample_indices:

            n_state = []
            n_action = []
            n_reward = []
            n_next_state = []
            n_done = []

            for n in range(num_of_step):
                exp = self.buffer[s + n]

                n_state.append(exp[0])
                n_action.append(exp[1])
                n_reward.append(exp[2])
                n_next_state.append(exp[3])
                n_done.append(exp[4])

            sample_data.append((n_state, n_action, n_reward, n_next_state, n_done))

        return sample_data