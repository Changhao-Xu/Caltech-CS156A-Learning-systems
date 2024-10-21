import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
import pandas as pd

def f(x1, x2):
    return np.sign(x2 - x1 + 0.25 * np.sin(np.pi * x1)) # target function

class kmeans_RBF:
    def __init__(self, num_clusters = None, gamma = None):
        '''
        - Takes number of clusters K
        - returns weight vector learned by regular RBF model,
        i.e. using Lloyd's algorithm + pseudo inverse
        '''
        self.K = num_clusters
        self.cluster_centers = None
        self.cluster_index_of_x = None
        self.w = None
        self.gamma = gamma

    def fit(self, X, y):
        N = X.shape[0]
        
        '''
        - Takes points X (numpy array)
        - Calculates final cluster centers
        - Calculates cluster index of each point x
        - Returns None
        '''
        while True:
            empty_cluster_detected = False
            in_sample_error_nonzero = False

            # We repeat the experiment until we get a case where all
            # clusters are non-empty

            # initialize centers by picking random points
            mu_list = np.random.uniform(-1,1,(self.K,2))
            
            #print("\ninitial centers: mu_list = ")
            #print(mu_list)
            
            #------------

            # cluster_of_x stores for each point x its cluster
            cluster_of_x = [-1 for _ in range(N)]
            old_cluster_of_x = [-1 for _ in range(N)]


            MAX_ITERATIONS = 10**6

            for i in range(MAX_ITERATIONS):

                # initialize clusters
                S = [[] for _ in range(self.K)]

                # assign each point to a cluster
                for point_index, x in enumerate(X):
                    # determine for each point its nearest cluster
                    min_distance = 2**64
                    min_cluster = None
                    for index, mu in enumerate(mu_list):
                        distance = np.linalg.norm(x - mu)
                        if distance < min_distance:
                            min_distance = distance
                            min_cluster = index
                    S[min_cluster].append(x)
                    cluster_of_x[point_index] = min_cluster

                # check if there is an empty cluster
                for cluster in S:
                    if not cluster:
                        #print("\nEmpty cluster detected, discarding run")
                        empty_cluster_detected = True

                if empty_cluster_detected:
                    break

                #----------------------------------

                # stop if nothing changes, i.e. points are in the same clusters as in previous iteration
                if cluster_of_x == old_cluster_of_x:
                    #print("Cluster have not changed, stopping for loop...")
                    break

                #------------------------------------------------------------------

                # make a copy
                old_cluster_of_x = [cluster_index for cluster_index in cluster_of_x]

                # calculate the new centers mu
                for index, cluster in enumerate(S):
                    mu = sum(cluster) / len(cluster)   # compute center of gravity
                    mu_list[index] = mu

            #print("\nfinal centers: mu_list = ")
            #print(mu_list)
            


            #if discard_run == False:
            #    break
            if (empty_cluster_detected == True):
                #print("\nEmpty cluster detected, discarding run")
                continue
            
                
            # setting attributes
            self.cluster_centers = mu_list
            self.cluster_index_of_x = cluster_of_x


            # calculate w via linear regression
            def matrix_phi_entry(x, mu, gamma):
                return np.exp(-gamma * np.linalg.norm(x - mu)**2)

            # initialize phi
            phi = np.zeros((N, self.K))

            # fill matrix phi
            for i in range(N):
                for k in range(self.K):
                    phi[i,k] = matrix_phi_entry(X[i], mu_list[k], self.gamma)

            phi = np.c_[np.ones(N), phi]

            #print("\nphi for training points = ")
            #print(phi)

            self.w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), y)   # coefficients w
            #print("\nw = ", self.w)
            
            
            #if (in_sample_error_nonzero == False):
            #    break
            break
                
        #-----------------------------------------------------------------------


    def predict(self, X_test):
        '''
        - Takes points X
        - Returns predicted y
        '''
        
        def matrix_phi_entry(x, mu, gamma):
            return np.exp(-self.gamma * np.linalg.norm(x - mu)**2)

        # initialize phi
        N_test = X_test.shape[0]
        phi = np.zeros((N_test, self.K))
        #print("shape of phi: ", phi.shape)

        # fill matrix phi
        for i in range(N_test):
            for k in range(self.K):
                phi[i,k] = matrix_phi_entry(X_test[i], self.cluster_centers[k], self.gamma)


        phi = np.c_[np.ones(N_test), phi]
        
        #print("\nphi matrix:")
        #print(phi)
        
        y_predicted = np.sign(np.dot(phi, self.w))
        return y_predicted

def count_kernel_better_than_regular_RBF(K, number_of_runs):
    count_svm_better_than_lloyd = 0
    
    for run_counter in np.arange(number_of_runs):
        N = 100
        X_train = np.random.uniform(-1,1,(N,2)) # generate 100 training points
        y_train = f(X_train[:,0], X_train[:,1])
        X_test = np.random.uniform(-1,1,(N,2))  # generate 100 testing points
        y_test = f(X_test[:,0], X_test[:,1])


        clf = svm.SVC(C = np.inf, kernel = 'rbf', gamma = 1.5) # hard-margin SVM with RBF kernel, gamma = 1.5
        clf.fit(X_train, y_train)
        E_out_svm = sum(clf.predict(X_test) != y_test) / N
        #print("E_out for SVM = ", E_out_svm)
        
        # compute in-sample error E_in for the SVM classifier
        E_in_svm = sum(clf.predict(X_train) != y_train) / N
        if E_in_svm > 0:
            print("E_in for SVM classifier is nonzero! Discarding run")
            continue

        # Lloyd + pseudoInv
        lloyd = kmeans_RBF(num_clusters = K, gamma = 1.5)
        lloyd.fit(X_train, y_train)
        E_out_lloyd = sum(lloyd.predict(X_test) != y_test) / N
        #print("E_out for Lloyd = ", E_out_lloyd)
    
        if E_out_svm < E_out_lloyd:
            count_svm_better_than_lloyd += 1
      
    print("SVM with RBF kernel beats regular RBF %f%% in terms of Eout" % (count_svm_better_than_lloyd / number_of_runs * 100))

def main():
    print("Q14 is based on the following results:")
    N_train = 100
    x1 = np.random.uniform(-1,1,N_train)
    x2 = np.random.uniform(-1,1,N_train)
    y_train = np.sign(x2 - x1 + 0.25 * np.sin(np.pi * x1))
    X_train = np.c_[x1, x2]
    # set up the classifier
    clf = svm.SVC(C = np.inf, kernel = 'rbf', gamma = 1.5)

    # train the classifier
    clf.fit(X_train, y_train)

    # predict on training points
    y_predict_train = clf.predict(X_train)

    # compute in-sample error E_in
    E_in = sum(y_predict_train != y_train) / N_train
    print("In-sample error E_in:", E_in)
    N_train = 100
    
    E_in_nonzero_counter = 0

    for run_counter in np.arange(100):
        
        # generate training data
        x1 = np.random.uniform(-1,1,N_train)
        x2 = np.random.uniform(-1,1,N_train)
        y_train = np.sign(x2 - x1 + 0.25 * np.sin(np.pi * x1))
        X_train = np.c_[x1, x2]

        clf = svm.SVC(C = np.inf, kernel = 'rbf', gamma = 1.5)
        clf.fit(X_train, y_train)

        # predict on training points
        y_predict_train = clf.predict(X_train)

        # compute in-sample error E_in
        E_in = sum(y_predict_train != y_train) / N_train
        if E_in > 0: E_in_nonzero_counter += 1

    print("That means E_in was nonzero %f%% of the time" % (E_in_nonzero_counter / 100 * 100))

    print("Q14 is based on the following results:")
    count_kernel_better_than_regular_RBF(9, 100)
    print("")

    print("Q15 is based on the following results:")
    count_kernel_better_than_regular_RBF(12, 100)
    print("")

    print("Q16 is based on the following results:")
    Ein_down = 0
    Ein_up = 0
    Eout_down = 0
    Eout_up = 0
    for run_counter in np.arange(100): # run 100 times
        N = 100
        X_train = np.random.uniform(-1,1,(N,2)) # generate 100 training points
        y_train = f(X_train[:,0], X_train[:,1])
        X_test = np.random.uniform(-1,1,(N,2))  # generate 100 testing points
        y_test = f(X_test[:,0], X_test[:,1])
        
        # for a fixed given data set, compare the two regular RBF
        E_in_values = [None, None]
        E_out_values = [None, None]
        
        for index, K in enumerate([9, 12]):
            # regular RBF = Lloyd + pseudoInv
            lloyd = kmeans_RBF(num_clusters = K, gamma = 1.5)
            lloyd.fit(X_train, y_train)
            E_in_lloyd = sum(lloyd.predict(X_train) != y_train) / N
            E_out_lloyd = sum(lloyd.predict(X_test) != y_test) / N
            
            E_in_values[index] = E_in_lloyd
            E_out_values[index] = E_out_lloyd

        if (E_in_values[0] > E_in_values[1]):
            Ein_down += 1
        elif (E_in_values[0] < E_in_values[1]):
            Ein_up += 1
        
        if (E_out_values[0] > E_out_values[1]):
            Eout_down += 1
        elif (E_out_values[0] < E_out_values[1]):
            Eout_up += 1

    if (Ein_down > Ein_up):
        print("%f%% time Ein goes down" % (Ein_down / (Ein_down + Ein_up) * 100))
    elif (Ein_down < Ein_up):
        print("%f%% time Ein goes up" % (Ein_up / (Ein_down + Ein_up) * 100))
    else:
        print("Ein remains the same")

    if (Eout_down > Eout_up):
        print("%f%% time Eout goes down" % (Eout_down/(Eout_down + Eout_up) * 100))
    elif (Eout_down < Eout_up):
        print("%f%% time Eout goes up" % (Eout_up/(Eout_down + Eout_up) * 100))
    else:
        print("Eout remains the same")

    print("")

    print("Q17 is based on the following results:")
    Ein_down = 0
    Ein_up = 0
    Eout_down = 0
    Eout_up = 0
    for run_counter in np.arange(100): # run 100 times
        N = 100
        X_train = np.random.uniform(-1,1,(N,2)) # generate 100 training points
        y_train = f(X_train[:,0], X_train[:,1])
        X_test = np.random.uniform(-1,1,(N,2))  # generate 100 testing points
        y_test = f(X_test[:,0], X_test[:,1])
        
        # for a fixed given data set, compare the two regular RBF
        E_in_values = [None, None]
        E_out_values = [None, None]
        
        for index, gamma_value in enumerate([1.5, 2]):

            # regular RBF = Lloyd + pseudoInv
            lloyd = kmeans_RBF(num_clusters = 9, gamma = gamma_value)
            lloyd.fit(X_train, y_train)
            E_in_lloyd = sum(lloyd.predict(X_train) != y_train) / N
            E_out_lloyd = sum(lloyd.predict(X_test) != y_test) / N
            E_in_values[index] = E_in_lloyd
            E_out_values[index] = E_out_lloyd

        if (E_in_values[0] > E_in_values[1]):
            Ein_down += 1
        elif (E_in_values[0] < E_in_values[1]):
            Ein_up += 1
        
        if (E_out_values[0] > E_out_values[1]):
            Eout_down += 1
        elif (E_out_values[0] < E_out_values[1]):
            Eout_up += 1

    if (Ein_down > Ein_up):
        print("%f%% time Ein goes down" % (Ein_down / (Ein_down + Ein_up) * 100))
    elif (Ein_down < Ein_up):
        print("%f%% time Ein goes up" % (Ein_up / (Ein_down + Ein_up) * 100))
    else:
        print("Ein remains the same")

    if (Eout_down > Eout_up):
        print("%f%% time Eout goes down" % (Eout_down/(Eout_down + Eout_up) * 100))
    elif (Eout_down < Eout_up):
        print("%f%% time Eout goes up" % (Eout_up/(Eout_down + Eout_up) * 100))
    else:
        print("Eout remains the same")

    print("")

    print("Q18 is based on the following results:")
    counter_E_in_equals_zero = 0
    for run_counter in np.arange(100): # run 100 times
        N = 100
        X_train = np.random.uniform(-1,1,(N,2)) # generate 100 training points
        y_train = f(X_train[:,0], X_train[:,1])
        X_test = np.random.uniform(-1,1,(N,2))  # generate 100 testing points
        y_test = f(X_test[:,0], X_test[:,1])

        
        # regular RBF = Lloyd + pseudoInv with K = 9 and gamma = 1.5
        lloyd = kmeans_RBF(num_clusters = 9, gamma = 1.5)
        lloyd.fit(X_train, y_train)
        E_in_lloyd = sum(lloyd.predict(X_train) != y_train) / N
        if E_in_lloyd == 0:
            counter_E_in_equals_zero += 1

    print("E_in is zero %f%% of the time" % (counter_E_in_equals_zero / 100 * 100))

if __name__== "__main__":
    main()