import gym
import numpy as np
import os
import torch
import argparse

from logger import Logger
from utils import stack_frames

parser = argparse.ArgumentParser()

parser.add_argument('--env-name', type=str, default='Pendulum-v1',
                    help='Environment name.')
parser.add_argument('--num-episodes', type=int, default=100,
                    help='Number of episodes.')
parser.add_argument('--observation-dim-w', type=int, default=84,
                    help='Width of the input measurements (RGB images).')
parser.add_argument('--observation-dim-h', type=int, default=84,
                    help='Height of the input measurements (RGB images).')
parser.add_argument('--test', default=False,
                    help='Generate training or testing dataset.')
parser.add_argument('--training-dataset', type=str, default='pendulum_train.pkl',
                    help='Training dataset.')
parser.add_argument('--testing-dataset', type=str, default='pendulum_test.pkl',
                    help='Testing dataset.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (default: 1).')
parser.add_argument('--random-policy', default=True,
                    help='Use random action policy.')

args = parser.parse_args()

env_name = args.env_name
test = args.test
if test:
    data_file_name = args.testing_dataset
else:
    data_file_name = args.training_dataset
obs_dim1 = args.observation_dim_w
obs_dim2 = args.observation_dim_h
num_episodes = args.num_episodes
seed = args.seed
random_policy = args.random_policy

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

directory = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(directory + '/Data/')
logger = Logger(folder)

# Set seeds
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

max_steps = 200 #default for pendulum-v1

for episode in range(num_episodes):
    state = env.reset()
    frame = np.array(env.render(mode='rgb_array'))
    prev_frame = np.array(env.render(mode='rgb_array'))
    print('Episode: ', episode)
    for step in range(max_steps):
        obs = stack_frames(prev_frame, frame, obs_dim1, obs_dim2)
        if random_policy:
            action = env.action_space.sample()
        else:
            pass
            # action = policy(observation)
        next_state, reward, done, info = env.step(action)
        next_frame = np.array(env.render(mode='rgb_array'))
        next_obs = stack_frames(frame, next_frame, obs_dim1, obs_dim2)

        if step == max_steps - 1:
            done = True

        logger.obslog((obs, action, reward, next_obs, done, state))
        prev_frame = frame
        frame = next_frame
        state = next_state

        if done:
            break

logger.save_obslog(filename=data_file_name)