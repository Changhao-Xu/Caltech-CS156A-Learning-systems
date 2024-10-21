import numpy as np

#want: X = x1, x2
class RegLinReg:
    def __init__(self, l_reg, nlt=False):
        self.l_reg = l_reg #lambda regularization term
        self.nlt = nlt == True #to use a nonlinear transform or not (for now, order 2)

    def set_lambda(self, l_reg):
        self.l_reg = l_reg

    def set_nlt(self, nlt):
        self.nlt = nlt == True

    def X_reshape(self, X):
        num_ex = X.shape[0]
        if self.nlt == False:
            #no transform
            X_res = np.c_[np.ones(num_ex), X]
        else:
            #nlt = (1, x1, x2, x1x2, x1^2, x2^2)
            X_mult = np.prod(X, axis=1)
            X_res = np.c_[np.ones(num_ex), X, X_mult, np.square(X)]
        return X_res
            
    def predict(self,X):
        real_X = self.X_reshape(X)
        cur_h = np.matmul(real_X, self.weights)
        return cur_h

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
        
        
 


    
