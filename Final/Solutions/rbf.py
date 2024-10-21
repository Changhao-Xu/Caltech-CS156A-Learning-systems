import numpy as np

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

        
        
        
 
