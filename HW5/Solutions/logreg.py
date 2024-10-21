import math
import numpy as np

class LogReg:
    def __init__(self,dim, l_rate):
        dim = max(1, dim)
        self.dim = dim
        self.l_rate = l_rate # learning rate
        self.weights = np.zeros(dim+1)

    def init_weights(self):
        self.weights = np.zeros(self.dim + 1)
        
    def reshape_X(self, X):
        #number of examples
        if len(X.shape) > 1 and X.shape[0] >= 1:
            num_ex = X.shape[0]
            return np.c_[ np.ones(num_ex), X]
        else:
            cur_size = X.size
            return np.r_[1, X]

    def risk_score(self, X):
        #should return (n, 1)
        res_X = self.reshape_X(X)
        my_risk = np.dot(res_X,self.weights)
        return my_risk

    def sigmoid(self, X):
        #theta(s) = e^s/(1+e^s)
        cur_es = np.exp(risk_score(X))
        return np.divide(cur_es, np.add(1, cur_es))

    def gradient(self, X, y):
        #grad(E_in) = (-1/N)*sum(n=1;N){(y_n*x_n)/(1+e^(y_n*wT(t)*x_n))}
        res_X = self.reshape_X(X)
        cur_N = X.shape[0]
        cur_numer = np.multiply(y,res_X) #y_n*x_n by row, should be (n,dim+1)
        #should return (n,1)
        cur_denom = np.add(1, np.exp(np.multiply(y, self.risk_score(X))))
        #divide cur_numer row wise by cur_denom, should still be (n, dim+1)
        presum = np.divide(cur_numer, cur_denom)
        #sum by column
        cur_sum = np.sum(presum, axis = 0)
        #now normalize by (-1/N) and return
        cur_sum = np.divide(cur_sum, -1*cur_N)
        return cur_sum

    def sto_gradient(self, xn, yn):
        #stochastic gradient, should be only one example
        res_X = self.reshape_X(xn)
        cur_numer = np.multiply(yn, res_X)
        cur_denom = np.add(1, np.exp(np.multiply(yn, self.risk_score(xn))))
        return np.multiply(-1, np.divide(cur_numer, cur_denom))
    
    def update_weights(self, X, y):
        #w(t+1) = w(t) - l_rate * gradient
        cur_grad = self.gradient(X,y)
        self.weights = np.subtract(self.weights, np.multiply(self.l_rate, cur_grad))
    
    def sto_gd(self, X, y):
        # a run of stochastic gradient descent
        cur_num = X.shape[0]
        #get indices for every row/example in X and shuffle them
        cur_idxs = np.arange(cur_num)
        np.random.shuffle(cur_idxs)
        #now update weights one by one
        for cur_idx in cur_idxs:
            cur_grad = self.sto_gradient(X[cur_idx], y[cur_idx])
            self.weights = np.subtract(self.weights, np.multiply(self.l_rate, cur_grad))

    def ce_error(self, X, y):
        #cross-entropy error
        #e_in = (1/N) sum(n=1;N){ ln(1+e^(-yn*wT*xn))}
        res_X = self.reshape_X(X)
        cur_N = res_X.shape[0]
        cur_val = np.log(np.add(1, np.exp(np.multiply(np.multiply(-1,y), self.risk_score(X)))))
        #should be (n,1)
        return np.divide(np.sum(cur_val), cur_N)
        

        

    
