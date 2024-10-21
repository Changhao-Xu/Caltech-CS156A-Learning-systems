import numpy as np

class PLA:
    def __init__(self, thresh):
        #threshold for pct wrong, for guaranteed linearly seperable, set to 0
        self.thresh = max(0, thresh)
        #since we don't want sit here forever, let's stick an upper limit to training rounds
        self.limit = 10e5
        

    def reset_weights(self):
        self.weights = np.zeros(1+self.dim) #adding one for w0

    def set_dim(self, dim):
        self.dim = max(1, dim)
        self.reset_weights()
        self.n_iter = 0 #number of training rounds
        
    def X_reshape(self,X):
        if len(X.shape) <= 1:
            return np.r_[[1.0], X]
        num_examples = X.shape[0]
        real_X = np.c_[np.ones(num_examples), X]
        return real_X

    def predict(self,X):
        real_X = self.X_reshape(X)
        cur_h = np.dot(real_X, self.weights)
        return cur_h
            

    def calc_error(self, X,Y):
        num_ex = X.shape[0]
        predicted = np.sign(self.predict(X))
        num_incorrect = np.sum(np.not_equal(predicted, np.sign(Y)))
        prop_incorrect = float(num_incorrect)/float(num_ex)
        return prop_incorrect

    def train(self,X,Y):
        if len(X.shape) <= 1:
            dim = X.shape[0]
            X = self.X_reshape((1,dim))
        self.set_dim(X.shape[1])
        num_ex = X.shape[0]
        ex_idxs = np.arange(num_ex) #indexing the number of examples and shuffling
        under_thresh = False
        n_iter = 0 #number of iterations
        while under_thresh == False:
            np.random.shuffle(ex_idxs)
            num_wrong = 0
            for idx in ex_idxs:
                guess = self.predict(X[idx])
                real_ex = np.r_[[1.0], X[idx]] #current example with x0
                agreed = np.sign(guess) == Y[idx]
                if not agreed:
                    self.weights = self.weights + np.multiply(Y[idx], real_ex)
                    num_wrong = num_wrong + 1
            pct_wrong = float(num_wrong)/float(num_ex)
            under_thresh = pct_wrong <= self.thresh
            n_iter = n_iter + 1
            if n_iter >= self.limit:
                break
        self.n_iter = n_iter
            
                
                
        
        
