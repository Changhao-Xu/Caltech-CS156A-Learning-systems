import numpy as np
from linreg import LinReg

# want: 1, x1, x2, x1^2, x2^2, x1*x2, abs(x1-x2), abs(x1+x2)
class LinRegNLT2(LinReg):
    def __init__(self, dim, k, l_reg):
        #want squares of both elements, both elts multiplied, then abs sub and add
        # = 2*dim + 3 
        self.dim = (2*dim + 3)
        #adding the x0 bit
        self.weights = np.zeros((self.dim + 1, 1))
        self.k = int(max(0,k)) # use columns up through k in nonlinear mapping (0-idx)
        self.l_reg = l_reg #lambda regularization term
        
    def set_lambda(self, l_reg):
        self.l_reg = l_reg

    def set_k(self, k):
        self.k = int(max(0,k))

    def X_reshape(self,X):
        #do the nonlinear transform here
        num_ex = X.shape[0] #number of examples
        X_mult = np.prod(X, axis=1)
        X_sub_mtx = np.c_[ X[:,0], np.multiply(-1, X[:,1:])] #subtraction matrix
        X_res = np.c_[np.ones(num_ex), X, np.square(X), X_mult, np.abs(np.sum(X_sub_mtx, axis=1)), np.abs(np.sum(X, axis=1))]
        return X_res[:,:(self.k + 1)]

    def calc_error(self, X,Y):
        num_ex = X.shape[0]
        predicted = np.sign(self.predict(X))
        num_incorrect = np.sum(np.not_equal(predicted, np.sign(Y)))
        prop_incorrect = float(num_incorrect)/float(num_ex)
        return prop_incorrect

    #training with regularization:
    # (ZT*Z + lambda*I)^-1 * ZT*y
    def train_reg(self, X,Y):
        X_res = self.X_reshape(X)
        xtx = np.dot(X_res.T, X_res)
        lm = np.multiply(self.l_reg, np.identity(xtx.shape[0])) #lambda*I
        X_inv = np.linalg.inv(np.add(xtx, lm))
        self.weights = np.dot(X_inv, np.dot(X_res.T, Y))
        
        
        
        
        
