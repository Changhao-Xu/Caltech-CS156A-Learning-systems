import numpy as np

class LFD_Data2:
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
                ret_X = np.concatenate((ret_X, cur_XY[1:])) #everything but first elt
                ret_Y = np.concatenate((ret_Y, [cur_XY[0]])) #first elt
        ret_X = ret_X.reshape((num_ex, X_dim))
        self.dim = X_dim
        return ret_X, ret_Y
            
    def __init__(self, trainfile, testfile):
        self.dim = 0
        self.train_X, self.train_Y = self.load_file(trainfile)
        self.test_X, self.test_Y = self.load_file(testfile)
        self.filt_argc = 0

    def Y_mapper(self, Y, eql):
        #maps elts equal to eql to 1 else -1
        return np.subtract(np.multiply(2, np.equal(Y.astype(int), int(eql)).astype(int)), 1)

    def filt_idx(self, Y):
        #returns filtered indices according to my_filt
        return np.where(self.my_filt(Y))[0]

        
    def set_filter(self, params=[]):
        #0 args: no filter
        #1 arg: 1-vs-all - desired digit gets 1, else -1
        #2 args: 1-vs-1 - first digit gets 1, other gets -1, else omitted
        self.filt_argc = min(2, len(params))
        self.filt_argv = params
        if len(params) == 2:
            self.my_filt = np.vectorize(lambda x: int(x) == params[0] or int(x) == params[1])

    def get_X(self, req_set = "train"):
        if req_set.lower() == "train".lower():
            if self.filt_argc == 0 or self.filt_argc == 1:
                return self.train_X
            elif self.filt_argc == 2:
                #filtered indices
                filtered = self.filt_idx(self.train_Y)
                return self.train_X[filtered]
        else:
            if self.filt_argc == 0 or self.filt_argc == 1:
                return self.test_X
            elif self.filt_argc == 2:
                #filtered indices
                filtered = self.filt_idx(self.test_Y)
                return self.test_X[filtered]

    def get_Y(self, req_set = "train"):
        if req_set.lower() == "train".lower():
            if self.filt_argc == 0:
                return self.train_Y
            elif self.filt_argc == 1:
                #one-liner for mapping given param as 1 else -1
                return self.Y_mapper(self.train_Y, self.filt_argv[0])
            elif self.filt_argc == 2:
                #filtered indices
                filtered = self.filt_idx(self.train_Y)
                return self.Y_mapper(self.train_Y[filtered], self.filt_argv[0])
        else:
            if self.filt_argc == 0:
                return self.test_Y
            elif self.filt_argc == 1:
                #one-liner for mapping given param as 1 else -1
                return self.Y_mapper(self.test_Y, self.filt_argv[0])
            elif self.filt_argc == 2:
                #filtered indices
                filtered = self.filt_idx(self.test_Y)
                return self.Y_mapper(self.test_Y[filtered], self.filt_argv[0])
                
                
                
                
                
        
        
