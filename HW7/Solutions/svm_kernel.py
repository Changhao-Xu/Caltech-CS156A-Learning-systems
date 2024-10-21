import numpy as np
import cvxopt as cvo

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


class SVM_Poly():
    def __init__(self, exponent = 1, upper_limit = 0):
        self.thresh = 1.0e-5
        self.exponent = exponent
        self.upper_limit = upper_limit
        #suppress output
        cvo.solvers.options['show_progress'] = False

    def set_exponent(self, Q):
        self.exponent = Q

    def set_upper_limit(self, C):
        self.upper_limit = max(0, C)

    def kernel_calc(self, X2):
        #X2 = inputs
        #polynomial kernel (1+xnT*xm)^Q
        kernel = np.power(np.add(1, np.dot(self.X,X2.T)), self.exponent)
        return kernel

    def get_constraints(self, num_ex):
        #soft margin
        if self.upper_limit > 0:
            #make constraints matrix G, h being passed number of examples
            #-alphas <= 0
            G1 = np.multiply(-1, np.eye(num_ex))
            #alphas <= c
            G2 = np.eye(num_ex)
            G = np.vstack((G1, G2))
            h1 = np.zeros(num_ex)
            h2 = np.ones(num_ex)*self.upper_limit
            h = np.hstack((h1, h2))
            return cvo.matrix(G), cvo.matrix(h)
        else:
            #hard margin
            G = cvo.matrix(np.multiply(-1, np.eye(num_ex)))
            # h = 0
            h = cvo.matrix(np.zeros(num_ex))
            return G, h
    def ayK(self, Xin):
        #get the value of sum(alpha_n > 0) {alpha_n * y_n * K(x_n, input)}
        k_calc = self.kernel_calc(Xin)
        pre_sum = np.multiply(self.alphas, np.multiply(self.Y, k_calc))
        post_sum = np.sum(pre_sum, axis=0)
        return post_sum
        
        
    def predict(self,Xin):
        post_sum = np.add(self.ayK(Xin), self.bias)
        return post_sum

    
    def calc_error(self, Xin,Yin):
        num_ex = Xin.shape[0]
        predicted = np.sign(self.predict(Xin))
        num_incorrect = np.sum(np.not_equal(predicted, np.sign(Yin)))
        prop_incorrect = float(num_incorrect)/float(num_ex)
        return prop_incorrect


    def train(self,X,Y):
        #expecting X as Nxd matrix and Y as a Nx1 matrix
        #note: no reshaping for X
        X = X.astype(float)
        Y = Y.astype(float)
        self.X = X
        num_ex, cur_dim = X.shape
        self.Y = Y.reshape((num_ex, 1))
        k_calc = self.kernel_calc(X)
        q = cvo.matrix(np.multiply(-1, np.ones((num_ex,1))))
        P = cvo.matrix(np.multiply(np.outer(Y, Y), k_calc))
        A = cvo.matrix(Y.reshape(1, num_ex), tc='d')
        b = cvo.matrix(0.0)
        G, h = self.get_constraints(num_ex)
        cvo_sol = cvo.solvers.qp(P,q,G,h,A,b)
        alphas = np.ravel(cvo_sol['x'])
        alphas_thresh = np.greater_equal(alphas,self.thresh)
        sv_idx = np.argmax(alphas_thresh)
        self.alphas = alphas.reshape((num_ex, 1))
        self.num_alphas = np.sum(alphas_thresh)
        self.bias = Y[sv_idx] - self.ayK(X[sv_idx])
        


class SVM_RBF(SVM_Poly):
    def __init__(self, gamma = 1, upper_limit = 0):
        self.thresh = 1.0e-5
        self.gamma = gamma
        self.upper_limit = upper_limit
        #suppress output
        cvo.solvers.options['show_progress'] = False


    def kernel_calc(self, Xin):
        if len(Xin.shape) == 1:
            Xin = Xin.reshape((1, Xin.shape[0]))
        cur_m = self.X.shape[0]
        cur_n = Xin.shape[0]
        ret = np.ndarray((cur_m, cur_n))
        if self.X.shape[1] == Xin.shape[1]:
            for i in range(cur_m):
                for j in range(cur_n):
                    ret[i][j] = np.exp(-1 * self.gamma * np.linalg.norm(self.X[i] - Xin[j]))
        if ret.shape[0] == 1 and ret.shape[1] == 1:
            return ret[0][0]
        else:
            return ret
