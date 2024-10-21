import cvxopt as cvo
import numpy as np

#using cvxopt notation, it takes minimizes x in the following equation:
# 0.5 * xT * P * x + qT * x with constrants G*x <= h, Ax = b

#in the case of our lagrangian, P(i,j) = yi*yj*xi.T*xj
#given that y is a Nx1 matrix, the y components are essentially the outer product y*y.T
# the x compnents are just in the matrix product x*x.T

#our constraints xare y.T*alpha = 0 and alpha => 0
# w Ax=b is the equality constraint, we must make our first constraint fit it
# b = 0, A = y where y should be a row vector
# since we have a greater than 0 constrant and cvxopt takes a less than constraint, we must make our constraint negative
# also we want EACH alpha to be greater than 0, thus dot alpha and identity
# h = 0 vector
# G = -1 * NxN identity times alpha where alpha should be a 1XN matrix


class SVM:
    def __init__(self):
        self.thresh = 1.0e-5
        #suppress output
        cvo.solvers.options['show_progress'] = False

    def kernel_calc(self, X):
        #kernel calculation
        return np.dot(X,X.T)

    def get_constraints(self, num_ex):
        #make constraints matrices G, h being passed number of examples
        #-alphas <= 0
        G = cvo.matrix(np.multiply(-1, np.eye(num_ex)))
        # h = 0
        h = cvo.matrix(np.zeros(num_ex))
        return G, h

    def X_reshape(self,X):
        num_examples = X.shape[0]
        real_X = np.c_[np.ones(num_examples), X]
        return real_X

    def calc_error(self, X,Y):
        num_ex = X.shape[0]
        predicted = np.sign(self.predict(X))
        num_incorrect = np.sum(np.not_equal(predicted, np.sign(Y)))
        prop_incorrect = float(num_incorrect)/float(num_ex)
        return prop_incorrect

    def predict(self,X):
        real_X = self.X_reshape(X)
        cur_h = np.matmul(real_X, self.weights)
        return cur_h

    def train(self,X,Y):
        #expecting X as Nxd matrix and Y as a Nx1 matrix
        #note: no reshaping for X
        X = X.astype(float)
        Y = Y.astype(float)
        num_ex, cur_dim = X.shape
        q = cvo.matrix(np.multiply(-1, np.ones((num_ex,1))))
        P = cvo.matrix(np.multiply(np.outer(Y, Y), self.kernel_calc(X)))
        A = cvo.matrix(Y.reshape(1, num_ex), tc='d')
        b = cvo.matrix(0.0)
        G, h = self.get_constraints(num_ex)
        cvo_sol = cvo.solvers.qp(P,q,G,h,A,b)
        alphas = np.ravel(cvo_sol['x'])
        #now to find the weight vector = sum(i=1,N) an*yn*xn
        yx = np.multiply(Y.reshape((num_ex, 1)),X)
        weights= np.sum(np.multiply(alphas.reshape(num_ex,1), yx), axis=0)
        #now we want to find the w0 term so pick an sv and solve
        #yn(wTxn + b) = 1
        #-> 1/yn = wTxn + b, (1/yn)-wTxn = b
        #find the sv with the corresponding alpha bigger than thresh
        alphas_thresh = np.greater_equal(alphas,self.thresh)
        sv_idx = np.argmax(alphas_thresh)
        wtxn = np.dot(weights, X[sv_idx])
        cur_b = (1.0/Y[sv_idx]) - wtxn
        self.weights = np.concatenate(([cur_b], weights))
        self.alphas = alphas
        self.num_alphas = np.sum(alphas_thresh)
        
        
        
        
        
        
        
