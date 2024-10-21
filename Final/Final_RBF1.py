import numpy as np

#class for generating dataset

class RbfFData:
    def __init__(self, n):
        #n = number of data points
        n = max(1, n)
        X = np.random.uniform(-1,1, (n, 2))
        X_sin = np.sin(np.multiply(np.pi, X[:,0]))
        X_oper = np.array([-1, 1, 0.25])
        X_res = np.c_[X, X_sin]
        self.X = X
        self.Y = np.sign(np.matmul(X_res, X_oper))

#lloyds algorithm

#iteratively minimize sum(k=1,k){sum(xn elt Sk) {||xn-mk||^2}} wrt mk,Sk
#where mk is the mth center and Sk is the kth cluster

#mk = (1/|Sk|) sum(xn elt Sk){xn}
#Sk = {xn : ||xn - mk|| <= all ||xn - ml||}


class Lloyd:
    def calc_cluster_centers(self):
        #return true if all clusters nonempty, else false
        nonempty = True # if all clusters nonempty 
        for k,cluster in enumerate(self.cluster):
            if len(cluster) <= 0:
                nonempty = False
                break
            else:
                cur_cluster = self.X[cluster]
                self.cluster_centers[k] = np.average(cur_cluster, axis=0)
        return nonempty
            
            
    def assign_clusters(self):
        #returns if cluster membership changed or not
        changed = False
        #hopefully X is two-dim or else this breaks
        #iterate over X
        for n, xn in enumerate(self.X):
            cur_cluster = self.X_cluster[n]
            dest_cluster = cur_cluster #cluster that current xn ends up in
            shortest_dist = np.linalg.norm(self.cluster_centers[cur_cluster]-xn) #dist of xn from current cluster
            #iterate over clusters
            for l, cluster in enumerate(self.cluster_centers):
                cur_dist = np.linalg.norm(cluster - xn) #dist of xn from iterated cluster
                if cur_dist < shortest_dist:
                    dest_cluster = l
                    shortest_dist = cur_dist
            if cur_cluster != dest_cluster:
                self.cluster[cur_cluster].remove(n)
                self.cluster[dest_cluster].append(n)
                self.X_cluster[n] = dest_cluster
                changed = True
        return changed
                
    
    def init_clusters(self):
        self.cluster = [[] for x in range(self.k)]
        self.cluster[0] = [x for x in range(self.X_n)] #stick all in the first cluster for now
        #listing of cluster membership by elts of X
        self.X_cluster = [0 for x in range(self.X_n)]
        #cluster centers
        self.cluster_centers = np.random.uniform(self.rng[0], self.rng[1], (self.k, self.X_dim))
        self.assign_clusters()

    def set_X(self,X):
        #X = dataset, should be m x n np array
        self.X = X
        self.X_n = X.shape[0]
        if len(X.shape) == 1:
            self.X_dim = 1
        else:
            self.X_dim = X.shape[1]
        self.init_clusters()


    def __init__(self, X, k, rng):
        #k = number of clusters
        self.k = max(1, int(k))
        #rng = range of allowed center coords as an array
        self.rng = rng
        self.set_X(X)


    def set_k(self,k):
        if k != self.k:
            self.k = max(1, int(k))
            self.init_clusters()

    def run(self):
        runs = 1 #number of runs executed
        while True:
            while True:
                nonempty = self.calc_cluster_centers()
                if nonempty == True:
                    break
                else:
                    self.init_clusters()
                    runs = runs + 1
            changed = self.assign_clusters()
            if changed == False:
                break
        return runs

#h(x) = sign (sum(n=1;N) {wn * exp (-gamma * ||x-muk||^2)} + b)
#elts of phi matrix = exp(-gamma ||xi-muj||^2)

#this will be with a bias term so we need to reshape phi
class RBF:
    def set_X(self, X):
        self.lloyd.set_X(X)
        self.lloyd.run()

    def set_Y(self, Y):
        self.Y = Y

    def set_k(self, k):
        self.k = k
        self.lloyd.set_k(k)
        self.lloyd.run()

    def set_gamma(self, g):
        self.gamma = g
 
    def kernel_calc(self, Xin):
        #calculates exp( - gamma * ||Xin - mu||^2)
        if len(Xin.shape) == 1:
            Xin = Xin.reshape((1, Xin.shape[0]))
        cur_m = Xin.shape[0]
        cur_n = self.lloyd.cluster_centers.shape[0]
        ret = np.ndarray((cur_m, cur_n))
        if Xin.shape[1] == self.lloyd.cluster_centers.shape[1]:
            for i in range(cur_m):
                for j in range(cur_n):
                    ret[i][j] = np.exp(-1 * self.gamma * np.linalg.norm(Xin[i] - self.lloyd.cluster_centers[j]))
        if ret.shape[0] == 1 and ret.shape[1] == 1:
            return ret[0][0]
        else:
            return ret
               
    def __init__(self, gamma, X, Y, k, rng):
        #k = k centers for anticipated lloyd's algo
        #rng - 2-dim array of anticipated range allowable 
        self.gamma = gamma
        self.k = k
        self.rng = rng
        self.lloyd = Lloyd(X, k, rng)
        self.lloyd.run()
        self.Y = Y

    def train(self):
        phi = self.kernel_calc(self.lloyd.X)
        phi_n = phi.shape[0]
        #reshaping to get bias term
        phi_res = np.c_[np.ones(phi_n), phi]
        phi_pinv = np.linalg.pinv(phi_res)
        weights = np.matmul(phi_pinv, self.Y)
        self.bias = weights[0]
        self.weights = weights[1:]

    def predict(self, Xin):
        k_calc = self.kernel_calc(Xin)
        w_k = np.multiply(self.weights, k_calc)
        wk_sum = np.add(np.sum(w_k, axis=1), self.bias)
        return wk_sum
            
    def calc_error(self, Xin,Yin):
        num_ex = Xin.shape[0]
        predicted = np.sign(self.predict(Xin))
        num_incorrect = np.sum(np.not_equal(predicted, np.sign(Yin)))
        prop_incorrect = float(num_incorrect)/float(num_ex)
        return prop_incorrect

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

n_train = 100
n_runs = 200
gamma = [1.5, 2]
thresh = 1.0e-100
my_svm = SVM_RBF(gamma[0])
rbf_k = [9, 12]
rbf_rng = [-1,1]

#gamma, k
ein_15_9 = np.array([])
ein_15_12 = np.array([])
ein_2_9 = np.array([])
eout_15_9 = np.array([])
eout_15_12 = np.array([])
eout_2_9 = np.array([])

#kernel beats regular in eout
svm_beats_rbf9 = np.array([])
svm_beats_rbf12 = np.array([])

#svm nonseparable (ein not 0)
svm_nonsep = np.array([])

#ein is 0 for rbf, gamma 1.5, k = 9
rbf9_ein0 = np.array([])

for run in range(n_runs):
    cur_train = RbfFData(n_train)
    cur_test = RbfFData(n_train)
    my_rbf = RBF(gamma[0], cur_train.X, cur_train.Y, rbf_k[0], rbf_rng)
    my_svm.train(cur_train.X, cur_train.Y)
    svm_ein = my_svm.calc_error(cur_train.X, cur_train.Y)
    svm_eout = my_svm.calc_error(cur_test.X, cur_test.Y)
    svm_nonsep = np.concatenate((svm_nonsep, [svm_ein > thresh]))
    for k in rbf_k:
        for g in gamma:
            my_rbf.set_k(k)
            my_rbf.set_gamma(g)
            my_rbf.train()
            if k == 9 and g == 1.5:
                cein = my_rbf.calc_error(cur_train.X, cur_train.Y)
                cout = my_rbf.calc_error(cur_test.X, cur_test.Y)
                ein_15_9 = np.concatenate((ein_15_9, [cein]))
                eout_15_9 = np.concatenate((eout_15_9, [cout]))
                svm_beats_rbf9 = np.concatenate((svm_beats_rbf9, [svm_eout < cout]))
            elif k == 12 and g == 1.5:
                cein = my_rbf.calc_error(cur_train.X, cur_train.Y)
                cout = my_rbf.calc_error(cur_test.X, cur_test.Y)
                ein_15_12 = np.concatenate((ein_15_12, [cein]))
                eout_15_12 = np.concatenate((eout_15_12, [cout]))
                svm_beats_rbf12 = np.concatenate((svm_beats_rbf12, [svm_eout < cout]))
            elif k == 9 and g == 2:
                cein = my_rbf.calc_error(cur_train.X, cur_train.Y)
                cout = my_rbf.calc_error(cur_test.X, cur_test.Y)
                ein_2_9 = np.concatenate((ein_2_9, [cein]))
                eout_2_9 = np.concatenate((eout_2_9, [cout]))
            
pct_svm_nonsep =  100.0 * np.sum(svm_nonsep)/float(n_runs)
pct_rbf9_loses = 100.0 * np.sum(svm_beats_rbf9)/float(n_runs)
pct_rbf12_loses = 100.0 * np.sum(svm_beats_rbf12)/float(n_runs)
pct_rbf9_ein0 = 100.0 * np.sum(rbf9_ein0)/float(n_runs)
avg_ein_15_9 = 100.0 * np.average(ein_15_9)
avg_eout_15_9 = 100.0 * np.average(eout_15_9)
avg_ein_15_12 = 100.0 * np.average(ein_15_12)
avg_eout_15_12 = 100.0 * np.average(eout_15_12)
avg_ein_2_9 = 100.0 * np.average(ein_2_9)
avg_eout_2_9 = 100.0 *np.average(eout_2_9)

print("gamma = 1.5 | SVM non-separable pct: %f%%" % pct_svm_nonsep)
print("gamma = 1.5, k = 9 | SVM beats reg RBF pct: %f%%" % pct_rbf9_loses)
print("gamma = 1.5, k = 12 | SVM beats reg RBF pct: %f%%" % pct_rbf12_loses)
print("gamma = 1.5, k = 9 | reg RBF E_in = 0 pct: %f%%" % pct_rbf9_ein0)
print("")
print("~~gamma = 1.5 k=9~~")
print("E_in: %f%%" % avg_ein_15_9)
print("E_out: %f%%" % avg_eout_15_9)
print("")
print("~~gamma = 1.5 k=12~~")
print("E_in: %f%%" % avg_ein_15_12)
print("E_out: %f%%" % avg_eout_15_12)
print("")
print("~~gamma = 2 k=9~~")
print("E_in: %f%%" % avg_ein_2_9)
print("E_out: %f%%" % avg_eout_2_9)