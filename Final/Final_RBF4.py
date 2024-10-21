import numpy as np
from sklearn import svm

def rb_func(x, mu, gamma):
    return np.exp(-gamma * np.linalg.norm(x - mu) ** 2)

def rb_matrix(X, mu, gamma, row, column):
    rb_matrix = np.zeros((row, column))
    for i in range(row):
        for k in range(column):
            rb_matrix[i,k] = rb_func(X[i], mu[k], gamma)
    return rb_matrix

class kmeans_RBF:
    def __init__(self, num_clusters = None, gamma = None):
        self.K = num_clusters # K = number of clusters
        self.mu = None
        self.cluster_x = None # for each point x the cluster it belongs to
        self.weights = None
        self.gamma = gamma

    def k_means(self, N_train, X_train):
        centers = np.random.uniform(-1,1,(self.K,2)) # initial random cluster centers
        cluster_empty = False
        cluster_of_x = -1 * np.ones(N_train)
        previous_cluster = -1 * np.ones(N_train) # differentiate no change from iteration to iteration
        for i in range(100000): # Lloyd, take maximum iterations as 100000
                S = [[] for _ in range(self.K)] # initialize clusters
                for index, x in enumerate(X_train): # for a given cluster center, assign all points to cluster
                    min_dist = np.inf
                    min_clst = -1
                    for mu_index, mu in enumerate(centers):
                        dist = np.linalg.norm(x - mu)
                        if dist < min_dist:
                            min_dist = dist
                            min_clst = mu_index
                    S[min_clst].append(x)
                    cluster_of_x[index] = min_clst

                for cluster in S:
                    if not cluster:
                        cluster_empty = True

                if cluster_empty or (cluster_of_x == previous_cluster).all():
                    break

                previous_cluster = cluster_of_x
                # [cluster_index for cluster_index in cluster_of_x]

                for index, cluster in enumerate(S):
                    mu = sum(cluster) / len(cluster)   # for a given cluster, assign new cluster centers
                    centers[index] = mu
        
        return centers, cluster_of_x, cluster_empty

    def fit(self, X_train, y):
        while True:
            N_train = X_train.shape[0]
            centers, cluster_of_x, cluster_empty = self.k_means(N_train, X_train) 
            if (cluster_empty == True): # if cluster empty, repeat iterations
                continue
            
            self.mu = centers
            self.cluster_x = cluster_of_x

            phi = rb_matrix(X_train, self.mu, self.gamma, N_train, self.K)
            phi_new = np.c_[np.ones(N_train), phi] # add 1st column for bias
            pinv_phi = np.linalg.pinv(phi_new) # pseudo inverse
            self.weights = np.dot(pinv_phi,y)
            break

    def predict(self, X_test): #Takes points X, Returns predicted y
        N_test = X_test.shape[0]
        phi = rb_matrix(X_test, self.mu, self.gamma, N_test, self.K)
        phi_new = np.c_[np.ones(N_test), phi] # add 1st column for bias
        y_predicted = np.sign(np.dot(phi_new, self.weights))
        return y_predicted

def f(x1, x2):
    return np.sign(x2 - x1 + 0.25 * np.sin(np.pi * x1)) # target function

def data_set(num_pts):
    x1 = np.random.uniform(-1,1,num_pts)
    x2 = np.random.uniform(-1,1,num_pts)
    X_train = np.c_[x1, x2]
    Y_train = f(x1, x2)
    return X_train, Y_train

def compare_svm_with_rbf(K, gammaa):
    count = 0
    for num_run in np.arange(1000): # run 1000 times to make sure stable
        X_train, Y_train = data_set(100) # each time generate 100 points
        X_test, Y_test = data_set(100) # each time generate 100 points
        clf = svm.SVC(C = np.inf, kernel = 'rbf', gamma = gammaa)
        clf.fit(X_train, Y_train)
        Ein_svm = sum(clf.predict(X_train) != Y_train) / 100
        if Ein_svm > 0:
            continue
        Eout_svm = sum(clf.predict(X_test) != Y_test) / 100
    
        rbf = kmeans_RBF(num_clusters = K, gamma = gammaa)
        rbf.fit(X_train, Y_train)
        Eout_rbf = sum(rbf.predict(X_test) != Y_test) / 100
    
        if Eout_svm < Eout_rbf:
            count += 1
    print("SVM with kernel form beats regular RBF %f%% in terms of Eout" % (count / 1000 * 100))

def compare_regular_rbf(K, gammaa):
    Ein_down = 0
    Ein_up = 0
    Eout_down = 0
    Eout_up = 0
    for num_run in np.arange(100): # run 100 times
        X_train, Y_train = data_set(100) # each time generate 100 points
        X_test, Y_test = data_set(100) # each time generate 100 points
        Ein = [None, None] #compare regular RBF with K = 9 and K = 12
        Eout = [None, None]
        for i in np.arange(2):
            rbf = kmeans_RBF(num_clusters = K[i], gamma = gammaa[i])
            rbf.fit(X_train, Y_train)
            Ein_rbf = sum(rbf.predict(X_train) != Y_train) / 100
            Eout_rbf = sum(rbf.predict(X_test) != Y_test) / 100                
            Ein[i] = Ein_rbf
            Eout[i] = Eout_rbf

        if (Ein[0] > Ein[1]):
            Ein_down += 1
        if (Ein[0] < Ein[1]):
            Ein_up += 1
        if (Eout[0] > Eout[1]):
            Eout_down += 1
        if (Eout[0] < Eout[1]):
            Eout_up += 1

    if (Ein_down > Ein_up):
        print("%f%% of the time Ein goes down" % (Ein_down / 100 * 100))
    elif (Ein_down < Ein_up):
        print("%f%% of the time Ein goes up" % (Ein_up / 100 * 100))
    else:
        print("Ein remains the same")

    if (Eout_down > Eout_up):
        print("%f%% of the time Eout goes down" % (Eout_down / 100 * 100))
    elif (Eout_down < Eout_up):
        print("%f%% of the time Eout goes up" % (Eout_up / 100 * 100))
    else:
        print("Eout remains the same")

    print("")

def main():
    print("Q13 is based on the following results:")
    Ein = np.array([])
    for num_run in np.arange(1000): # run 1000 times to make sure stable
        X_train, Y_train = data_set(100) # each time generate 100 points
        clf = svm.SVC(C = np.inf, kernel = 'rbf', gamma = 1.5)
        clf.fit(X_train, Y_train)
        Ein = np.append(Ein, sum(clf.predict(X_train) != Y_train) / 100)
    Ein_svm = sum(Ein[i] != 0 for i in np.arange(1000)) / 100
    print("Ein_svm != 0 for %f%% of the time" % (Ein_svm * 100))
    print("")

    print("Q14 is based on the following results:")
    compare_svm_with_rbf(9, 1.5)
    print("")

    print("Q15 is based on the following results:")
    compare_svm_with_rbf(12, 1.5)
    print("")

    print("Q16 is based on the following results:")
    compare_regular_rbf([9, 12], [1.5, 1.5])

    print("Q17 is based on the following results:")
    compare_regular_rbf([9, 9], [1.5, 2])

    print("Q18 is based on the following results:")
    count = 0
    for num_run in np.arange(1000): # run 1000 times to make sure stable
        X_train, Y_train = data_set(100) # each time generate 100 points
        X_test, Y_test = data_set(100) # each time generate 100 points 
        rbf = kmeans_RBF(num_clusters = 9, gamma = 1.5)
        rbf.fit(X_train, Y_train)
        if sum(rbf.predict(X_train) != Y_train) / 100 == 0:
            count += 1
    print("Ein_rbf = 0 for %f%% of the time" % (count / 1000 * 100))

if __name__== "__main__":
    main()