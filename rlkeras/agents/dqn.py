import gym
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Layer, Dense
from tensorflow.keras.optimizers import Adam, SGD

from rlkeras.utils.memory import RandomReplayBuffer


class DQNAgents():

    def __init__(self, model, policy, replay_memory_size=10000):
        self.model = model
        self.policy = policy
        self.replay_buffer = RandomReplayBuffer(replay_memory_size)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def compile(self, optimizer=SGD(), loss='mse'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, env, num_of_episodes, num_of_max_steps_per_episode=None,
              discount_factor=.99, batch_size=32, visualize=True):
        """Implementation for Deep Q Learning agents trainer

`       With an GYM enviroment, traing the agent to play the game.

        # Arguments
            env:                            OpenAI gym enviroment
            num_of_episodes:                The number of episode to train
            num_of_max_steps_per_episode:   To limit the maximum step per episode
            discount_factor:                Discount for future step
            batch_size:                     The size of minibatch to sample from experience replay
            visualize:                      Disable rendering can speed up the training a little bit
        """

        # Obtain the size of state and action
        env_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        total_step = 0
        total_episode = 0;

        # Episode Loop
        for episode in range(num_of_episodes):

            state = env.reset()
            step = 1
            total_reward = 0;

            # Step loop
            while True:

                # Render grahpic on user request
                if visualize == True:
                    env.render()

                # First sample an action an interacte with the environment
                Q = self.model.predict(np.array(state).reshape(-1, env_space))

                # Select an action according to the predicted Q value
                action = self.policy.select_action(Q)

                # With the action, interact with the environment
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # Add experience to replay buffer
                self.replay_buffer.append(state, action, reward, next_state, done)

                if total_step > batch_size:
                    # Sample random minibatch of transition from replay buffer
                    minibatch = self.replay_buffer.sample(batch_size)

                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)

                    state_batch = np.array(state_batch)
                    next_state_batch = np.array(next_state_batch)
                    done_batch = np.array(done_batch)

                    # Calculate the Q value and Q target (Name the variable according to the DQN2013 paper)
                    Q_forward = self.model.predict_on_batch(next_state_batch)
                    Q_target = self.model.predict_on_batch(state_batch)

                    Q_target[range(batch_size), action_batch] = reward_batch + (1. - done_batch) * discount_factor * np.amax(Q_forward, axis=1)

                    # Perform gradient descent between Q target and Q predicted
                    self.model.train_on_batch(state_batch, Q_target)

                # Termination state
                if num_of_max_steps_per_episode is not None and step > num_of_max_steps_per_episode:
                    print("Episode OVER finished after {} timesteps".format(step))
                    break;

                if done:
                    print("Episode ({}/{}) finished after {} timesteps, total reward: {}".format(total_episode, num_of_episodes, step, total_reward))
                    break

                # Update statistic variable
                state = next_state
                step += 1

            total_step += step
            total_episode += 1

        print("Training completed. Total {} episode in {} timesteps".format(total_episode, total_step))

        return

    def test(self, env, num_of_episodes, num_of_max_steps_per_episode=None, visualize=True):
        """Implementation for Deep Q Learning agents tester

`       With a trained model, see how good is it performaning in a game

        # Arguments
            env:                            OpenAI gym enviroment
            num_of_episodes:                The number of episode to play
            num_of_max_steps_per_episode:   To limit the maximum step per episode
            visualize:                      Disable rendering can speed up the training a little bit
        """

        env_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        total_step = 0
        total_episode = 0;

        # Episode
        for episode in range(num_of_episodes):

            state = env.reset()
            step = 1
            total_reward = 0;

            # Step in episode
            while True:

                # Render grahpic on user request
                if visualize == True:
                    env.render()

                # First sample an action an interacte with the environment
                Q = self.model.predict(np.array(state).reshape(-1, env_space))

                # Select an action according to the predicted Q value
                action = self.policy.select_action(Q)

                # With the action, interact with the environment
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # Termination state
                if num_of_max_steps_per_episode is not None and step > num_of_max_steps_per_episode:
                    print("Episode OVER finished after {} timesteps".format(step))
                    break;

                if done:
                    print("Episode ({}/{}) finished after {} timesteps, total reward: {}".format(total_episode, num_of_episodes, step, total_reward))
                    break

                # Update statistic variable
                state = next_state
                step += 1

            total_step += step
            total_episode += 1

        print("Test completed. Total {} episode in {} timesteps".format(total_episode, total_step))

        return