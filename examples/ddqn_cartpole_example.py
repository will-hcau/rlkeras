import argparse
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

from rlkeras.agents.dqn import DQNAgents
from rlkeras.utils.policy import BoltzmannQPolicy


ENV_NAME = 'CartPole-v0'

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
args = parser.parse_args()


# Load env
env = gym.make(ENV_NAME)

nb_actions = env.action_space.n


# Build a simple fully connected network
model = Sequential()
model.add(Flatten(input_shape=env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


# Select policy
policy = BoltzmannQPolicy()

# Create Agents
dqn = DQNAgents(model, policy=policy)

# Compile the model with optimizer and loss function
dqn.compile(optimizer=Adam(lr=1e-3))


if args.mode == 'train':
	dqn.train(env, num_of_episodes=200, batch_size=32, enable_double_q=True, visualize=False)
	dqn.save_weights('dqn_{}_weight.h5f'.format(ENV_NAME), overwrite=True)

elif args.mode == 'test':
	dqn.load_weights('dqn_{}_weight.h5f'.format(ENV_NAME))
	dqn.test(env, num_of_episodes=10, visualize=True)


# Release env
env.close()