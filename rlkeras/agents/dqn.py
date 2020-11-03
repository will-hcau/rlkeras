import gym
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Lambda, Input, Layer, Dense
from tensorflow.keras.optimizers import Adam, SGD

from rlkeras.utils.memory import RandomReplayBuffer


class DQNAgents():

    def __init__(self, model, policy, enable_double_q=False, enable_dueling_network=False, multi_step=1, replay_memory_size=10000):
        """Implementation for Deep Q Learning agents 

`       With an GYM enviroment, train the agent to play the game.

        # Arguments
            mode:                   The neural network model contructed by keras
            policy:                 The policy to pick action during the training
            enable_double_q:        Enable the double Q feature
            enable_dueling_network: Enable the dueling network feature
            multi_step:             Use N-step Q learning for training
            replay_memory_size:     The size of experience replay buffer
        """

        # Model Related
        self.model = model

        # Alogirthm related
        self.policy = policy
        self.replay_buffer = RandomReplayBuffer(replay_memory_size)

        # Double Q feature
        self.enable_double_q = enable_double_q

        # Multi-step feature
        self.multi_step = multi_step

        # Rebuild network architecture for dueling network
        if enable_dueling_network == True:

            # Cut off the two layer at the end and record the number of action space at the last layer
            layer = self.model.layers[-3]
            nb_action = self.model.output.shape[-1]

            # Add a fully connected layer with 1 more output respresenting A()
            dueling_layer = Dense(nb_action + 1, activation='linear')(layer.output)

            # Add lambda layer which add A() and the normalizied V()
            # This is a bit complicated presenting on code, please refer to "https://arxiv.org/abs/1511.06581"
            # for the network architecture visulization
            # ----------------------------------------------------------------------------------------------------------------------
            #                                   | -------Here is A()-------|   |------------And the normalizied V()-----------------|
            dueling_summation_layer = lambda dl: K.expand_dims(dl[:, 0], -1) + (dl[:, 1:] - K.mean(dl[:, 1:], axis=1, keepdims=True))

            output_layer = Lambda(dueling_summation_layer, output_shape=(nb_action,))(dueling_layer)

            self.model = Model(inputs=self.model.input, outputs=output_layer)

        # Clone a target model
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # Print network summary
        print(model.summary())

    def _recursive_bellman_equation(self, index, reward, done, discount_factor, max_q_t_plus_n):

        # Recall the bellman equation Q(st, at) = rt + discount * MAX Q(st+1, at + 1)
        # When comes to a N-step Q learning, it become a recusive function:
        # Q(st, at) = rt + discount * MAX Q(rt+1 + discount * MAX Q(rt + 2, at + 2))
        if (index >= self.multi_step - 1):
            return reward[:, index] + (1. - done[:, index]) * discount_factor * max_q_t_plus_n

        index += 1

        return reward[:, index] + (1. - done[:, index]) * discount_factor * self._recursive_bellman_equation(index, reward, done, discount_factor, max_q_t_plus_n)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def compile(self, optimizer=SGD(), loss='mse'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, env, num_of_episodes, num_of_max_steps_per_episode=None,
              target_model_update=1, discount_factor=.99, batch_size=32, visualize=True):
        """Implementation for Deep Q Learning agents trainer

`       With an GYM enviroment, traing the agent to play the game.
        Reference: "Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013"
                   "Human-level control through deep reinforcement learning, Mnih et al., 2015"

        # Arguments
            env:                            OpenAI gym enviroment
            num_of_episodes:                The number of episode to train
            num_of_max_steps_per_episode:   To limit the maximum step per episode
            target_model_update:            The update frequency of the target Q network
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
                    minibatch = self.replay_buffer.sample(batch_size, self.multi_step)

                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)

                    # Obtain the N-step item: [:,0] refer to 't' and [:,-1] refer to 't+N'
                    state_batch = np.array(state_batch)[:,0]
                    action_batch = np.array(action_batch)[:,0]
                    next_state_batch = np.array(next_state_batch)[:,-1]

                    done_batch = np.array(done_batch)
                    reward_batch = np.array(reward_batch)

                    if self.enable_double_q == True:
                        # Please refer to the Double deep Q learning paper on 2015
                        Q_forward = self.target_model.predict_on_batch(next_state_batch)
                        Q_target = self.model.predict_on_batch(state_batch)

                        # The tricky part is that to select action wusing the online network
                        # While the target network predict the estmated Q value so to avoid
                        # over estimating the Q value.
                        action = np.argmax(self.model.predict_on_batch(next_state_batch), axis=-1)

                        Q_target[range(batch_size), action_batch] = self._recursive_bellman_equation(0, reward_batch, done_batch, discount_factor, Q_forward[np.arange(len(Q_forward)), action])
                    else:
                        # Calculate the Q value and Q target (Name the variable according to the DQN2013 paper)
                        Q_forward = self.target_model.predict_on_batch(next_state_batch)
                        Q_target = self.model.predict_on_batch(state_batch)

                        # With the estimated Q value, update the Q target we aimed.
                        Q_target[range(batch_size), action_batch] = self._recursive_bellman_equation(0, reward_batch, done_batch, discount_factor, np.amax(Q_forward, axis=1))

                    # Perform gradient descent between Q target and Q predicted
                    self.model.train_on_batch(state_batch, Q_target)

                # The target Q (Please refer to the 2015 DQN paper)
                if (total_step + step) % target_model_update == 0:
                    self.target_model.set_weights(self.model.get_weights())

                # Termination state
                if num_of_max_steps_per_episode is not None and step > num_of_max_steps_per_episode:
                    print("Episode OVER finished after {} timesteps".format(step))
                    break;

                if done:
                    print("Episode ({}/{}) finished after {} timesteps, total reward: {}".format(total_episode+1, num_of_episodes, step, total_reward))
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