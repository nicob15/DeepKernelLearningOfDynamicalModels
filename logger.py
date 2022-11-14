import pickle
import os

class Logger():
    obsLog = []

    def __init__(self, folder, filename='dataset.pkl'):
        self.logDict = {}  # this log file is unique

        self.directory = folder
        self.filename = filename
        # make sure the folder exists
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def obslog(self, obs):
        self.obsLog.append(obs)

    def save_obslog(self, filename='dataset.pkl', folder=''):
        if folder == '':
            folder = self.directory
        with open(folder + filename, 'wb') as f:
            pickle.dump(self.obsLog, f, pickle.HIGHEST_PROTOCOL)
