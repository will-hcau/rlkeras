import argparse
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

from rlkeras.agents.dqn import DQNAgents
from rlkeras.utils.policy import BoltzmannQPolicy, GreedyQPolicy


ENV_NAME = 'LunarLander-v2'

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
args = parser.parse_args()


# Load env
env = gym.make(ENV_NAME)

print(env.action_space)
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


# Start training or testing
if args.mode == 'train':
	dqn = DQNAgents(model, enable_dueling_network=True, enable_double_q=True, multi_step=10, policy=BoltzmannQPolicy())
	dqn.compile(optimizer=Adam(lr=1e-3))
	dqn.train(env, num_of_episodes=500, batch_size=64, target_model_update=3000, visualize=False)
	dqn.save_weights('ddqn_{}_weight.h5f'.format(ENV_NAME), overwrite=True)

elif args.mode == 'test':
	dqn = DQNAgents(model, enable_dueling_network=True, policy=GreedyQPolicy())
	dqn.load_weights('ddqn_{}_weight.h5f'.format(ENV_NAME))
	dqn.test(env, num_of_episodes=10, visualize=True)


# Release env
env.close()