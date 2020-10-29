from collections import deque
import random

class RandomReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""
    def __init__(self, buffer_size):
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

    def sample(self, batch_size):
        """  Sampling

        Random sample a minibatch from the replay buffer

        """
        return random.sample(self.buffer, batch_size)