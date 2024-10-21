import numpy as np

class LFD_Data:
    def load_file(self, filename):
        ret_X = np.array([])
        ret_Y = np.array([])
        num_ex = 0 #number of examples
        X_dim = 0 #dimension of data
        with open(filename) as f:
            data = f.readlines()
            num_ex = len(data)
            X_dim = len(data[0].split()) - 1
            for line in data:
                cur_XY = [float(x) for x in line.split()]
                ret_X = np.concatenate((ret_X, cur_XY[:-1])) #everything but last elt
                ret_Y = np.concatenate((ret_Y, [cur_XY[-1]])) #last elt
        ret_X = ret_X.reshape((num_ex, X_dim))
        self.dim = X_dim
        return ret_X, ret_Y
            
    def __init__(self, trainfile, testfile):
        self.dim = 0
        self.train_X, self.train_Y = self.load_file(trainfile)
        self.test_X, self.test_Y = self.load_file(testfile)
