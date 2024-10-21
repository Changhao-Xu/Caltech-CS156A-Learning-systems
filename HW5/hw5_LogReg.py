import numpy as np

LR_EXP = 100 #number of times to run experiment
LR_N = 100 #number of points for training set, use for E_out also
LR_WTHRESH = 0.01 #desired weight change threshold

class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        diff = np.subtract(p2, p1)
        if diff[0] <= 0.0001: # if x_p1 == x_p2, then vertical
            self.k = None
            self.vertical = 1
            self.b = self.p1[0]
        else:
            self.k = diff[1]/diff[0] # k = (y_p2 - y_p1)/(x_p2 - x_p1)
            self.vertical = 0
            self.b = p1[1] - (self.k * p1[0]) # b = y_p1 - k * x_p1
        
    def compare(self,testpt):
        if self.vertical == 0:
            vector_y = self.k * testpt[0] + self.b
            diff = testpt[1] - vector_y
        else:
            vector_x = self.b # x = b
            diff = testpt[0] - vector_x
        return np.sign(diff)
    
    def calc_pts(self, ptset):
        #batch calculate points
        pt_dim = ptset.shape[1]
        my_calc = np.array([])
        for pt in ptset:
            cur_calc = self.compare(pt)
            my_calc = np.concatenate((my_calc, [cur_calc]))
        return my_calc

class LogReg:
    def __init__(self,dim, l_rate):
        dim = max(1, dim)
        self.dim = dim
        self.l_rate = l_rate # learning rate
        self.weights = np.zeros(self.dim + 1)

    def init_weights(self):
        self.weights = np.zeros(self.dim + 1)
        
    def reshape_X(self, X):
        num_ex = X.shape[0]
        return np.c_[ np.ones(num_ex), X] # insert 1 before each row of X

    def risk_score(self, X):
        #should return (n, 1)
        res_X = self.reshape_X(X)
        my_risk = np.dot(res_X,self.weights)
        return my_risk # calculate s = w_T * x

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
    
    def update_weights(self, X, y):
        #w(t+1) = w(t) - l_rate * gradient
        cur_grad = self.gradient(X,y)
        self.weights = np.subtract(self.weights, np.multiply(self.l_rate, cur_grad))

    def sto_gradient(self, xn, yn):
        #stochastic gradient, should be only one example
        res_X = self.reshape_X(xn)
        cur_numer = np.multiply(yn, res_X)
        cur_denom = np.add(1, np.exp(np.multiply(yn, self.risk_score(xn))))
        return np.multiply(-1, np.divide(cur_numer, cur_denom))
    
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
     

cur_logreg = LogReg(2, 0.01) #new logreg class with dim = 2 and learning rate 0.01
lr_epochs = np.array([]) #epoch record keeping
lr_eout = np.array([]) #e_out record keeping
for i in range(LR_EXP):
    cur_logreg.init_weights() #reset weights to 0
    cur_lpts = np.random.uniform(-1, 1, (2,2)) #generate points for line
    cur_line = Line(cur_lpts[0], cur_lpts[1])
    cur_train = np.random.uniform(-1,1,(LR_N,2)) #generate training set
    cur_labels = cur_line.calc_pts(cur_train) #get labels
    cur_epochs = 0 #init number of epochs
    cur_wdiff = 100 #init weight difference
    while cur_wdiff >= LR_WTHRESH:
        cur_epochs = cur_epochs + 1
        w_t = cur_logreg.weights #weights before training
        cur_logreg.sto_gd(cur_train, cur_labels) #run stochastic gradient descent which randomizes order of entries
        w_tp1 = cur_logreg.weights #weights after training
        cur_wdiff = np.linalg.norm(np.subtract(w_tp1, w_t)) #condition
    lr_epochs = np.concatenate(lr_epochs, [cur_epochs]) 
    cur_oos = np.random.uniform(-1,1, (LR_N, 2)) #out of sample testing
    oos_labels = cur_line.calc_pts(cur_oos)
    cur_eout = cur_logreg.ce_error(cur_oos, oos_labels)
    lr_eout = np.concatenate((lr_eout,[cur_eout]))

lr_epochs_avg = np.average(lr_epochs)
lr_eout_avg = np.average(lr_eout)


print("To converge on N=%d training examples:" % LR_N)
print("it took logistic regression an average %f epochs with an average E_out of %f" % (lr_epochs_avg, lr_eout_avg))