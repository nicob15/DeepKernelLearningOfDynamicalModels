import numpy as np
import torch
try:
    import cPickle as pickle
except:
    import pickle

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device))

def flatten_list(l):
    return [item for sublist in l for item in sublist]


def reshape_observation(observation):
    """
    reshape observations coming from the ROS/Gazebo simulations. The observation is composed of 40 lidar data points,
    goal position (xg, yg), robot pose (x, y theta) and 1 RGB image
    :param observation:
    :return: reshaped observation
    """
    observation = np.array(observation)
    obs_laser = np.array(observation[:40] * 100).astype(np.uint8)
    obs_laser = obs_laser.reshape(1,len(obs_laser))
    obs_camera = np.array(observation).astype(np.uint8)[45:]
    obs_camera = obs_camera.reshape(1,len(obs_camera))
    observation = np.concatenate((obs_camera, obs_laser), axis=1).reshape(1, len(observation) - 5).astype(np.float32) # the goal position (xg, yg) and the robot pose (x, y theta) are discarted as assumed not available
    return observation

def reshape_action(action):
    action = np.array(action)
    action = action.reshape((1,))
    action = np.column_stack((action, action.shape))

    return action

def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file):
    if not file[-3:] == 'pkl' and not file[-3:] == 'kle':
        file = file+'pkl'

    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data

def logvar2var(log_var):
    return torch.clip(torch.exp(log_var), min=1e-5)

def add_gaussian_noise(data, noise_level=0.0, clip=False, clip_level=(0, 1)):
    if clip:
        return (data + np.random.normal(0.0, noise_level, size=data.shape)).clip(clip_level[0], clip_level[1])
    else:
        return data + np.random.normal(0.0, noise_level, size=data.shape)

def add_square_distractor(data, size_1=10, size_2=10):
    distractor = np.zeros((size_1,size_2))
    x_dist = np.random.randint(0, data.shape[0] - size_1)
    y_dist = np.random.randint(0, data.shape[1] - size_2)
    if np.random.random() > 0.7:
        for i in range(size_1):
            for j in range(size_2):
                data[i+x_dist][j+y_dist][:] = distractor[i][j]
    return data