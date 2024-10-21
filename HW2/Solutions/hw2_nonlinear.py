from hw2_linreg import LRtest
from linreg import LinReg

import numpy as np

#new linreg class to deal with the special case of nonlineartransformation as given
#want:  x1, x2, x1*x2, x1^2, x2^2

#suppose this one is a bit harder to generalize than LinReg...
#mb just assume transformed number of columns  is (dim+1)*2, where the extra stuff
# comes from the matrix squared, the offset, and a col where all terms are multiplied
# not including offset dim
class LinRegNLT(LinReg):
    def __init__(self, dim):
        #super().__init__(dim)
        self.dim = (dim+1)*2 - 1 #minus offset col
        self.weights = np.zeros((self.dim + 1, 1))

    def X_reshape(self,X):
        #do the nonlinear transformation here
        num_examples = X.shape[0]
        X_mult = np.prod(X, axis=1) #col with mult across cols
        real_X = np.c_[np.ones(num_examples), X, X_mult, np.square(X)]
        return real_X

    


#desired target: sign(x1^2 + x2^2 - 0.6)


class NLTarget:
    def __init__(self,coeffs):
        #in this case, coeffs would be np.array([-0.6,1,1])
        #input input np.array([x1,x2])
        self.coeffs = coeffs

    def X_reshape(self,X):
        #add a ones col and square existing entries
        num_examples = X.shape[0]
        real_X = np.c_[np.ones(num_examples), np.square(X)]
        return real_X
        
    def calc(self, X):
        #can afford to do this matrix wise
        real_X = self.X_reshape(X) #add ones to left
        real_X = np.multiply(self.coeffs, real_X)
        sum_X = np.sum(real_X, axis=1)
        return np.sign(sum_X)
        
#desired noise amt: 10%

class NLTtest():
    #adding noise to labels
    def add_noise(self):
        amt = self.noise
        # number of indices to flip labels
        n_flip = int(self.n * amt)
        # label-flipping array consisting of
        # 1: don't flip
        # -1: flip
        flip_elts = np.multiply(-1, np.ones(n_flip))
        flip_arr = np.r_[np.ones(self.n - n_flip), flip_elts]
        np.random.shuffle(flip_arr)
        self.noisy_labels = np.multiply(flip_arr, self.labels)
 
    def __init__(self, numpoints, noise, coeffs):
        self.n = numpoints
        self.points = np.random.uniform(-1.0,1.0,(self.n, 2))
        self.noise = max(min(1, noise), 0)
        self.target = NLTarget(coeffs)
        self.lr = LinReg(2)
        self.nlt = LinRegNLT(2)
        self.labels = self.target.calc(self.points)
        self.add_noise()

    def regen_points(self, numpoints, amt):
        self.n = numpoints
        self.noise = max(min(1, amt), 0)
        self.points = np.random.uniform(-1.0,1.0,(self.n, 2))
        self.labels = self.target.calc(self.points)
        self.add_noise()

    def lr_train(self):
        self.lr.train(self.points, self.noisy_labels)
        
    def e_in(self, xw):
        xw = np.sign(xw)
        mydiff = np.not_equal(xw, self.noisy_labels)
        e_in = mydiff.mean()
        return e_in
       

    def lr_ein(self):
        xw = self.lr.predict(self.points)
        e_in = self.e_in(xw)
        return e_in

    def nlt_train(self):
        self.nlt.train(self.points, self.noisy_labels)

    def nlt_ein(self):
        xw = self.nlt.predict(self.points)
        e_in = self.e_in(xw)
        return e_in
        
        
numpts = 1000
noise_amt = 0.1
coeffs = np.array([-0.6, 1, 1])
numweights = 100
#want: first run with normal lin reg and find avg e_in
# use nonlinear transformed linreg and find w
# find e_out for nlt_linreg by generating 1000 new points and finding its e_in
def prob(num_exp):
    lr_ein = np.array([])
    nltlr_w = np.array([])
    nltlr_eout = np.array([])
    cur_nlt = NLTtest(numpts, noise_amt, coeffs)
    for i in range(num_exp):
        cur_nlt.regen_points(numpts, noise_amt)
        cur_nlt.lr_train()
        cur_lrein = cur_nlt.lr_ein()
        lr_ein = np.concatenate((lr_ein,[cur_lrein]))
        cur_nlt.nlt_train()
        cur_nltlrw = cur_nlt.nlt.weights
        if i < numweights:
            nltlr_w = np.concatenate((nltlr_w,cur_nltlrw))
        cur_nlt.regen_points(numpts, noise_amt)
        cur_nltlreout = cur_nlt.nlt_ein()
        nltlr_eout = np.concatenate((nltlr_eout,[cur_nltlreout]))
        """
        print(i)
        print("===")
        print(cur_nlt.lr.weights)
        print(cur_lrein)
        print(cur_nltlrw)
        print(cur_nltlreout)
        print("")
        """
    nltlr_w = nltlr_w.reshape(numweights, 6)
    avg_lr_ein = np.average(lr_ein)
    avg_nlt_w = np.average(nltlr_w, axis=0)
    avg_nlt_eout = np.average(nltlr_eout)
    print("average e_in from linear regression: %f" % avg_lr_ein)
    print("average weights from nlt-linear regression:")
    print(avg_nlt_w)
    print("")
    print("average e_out from nlt-linear regression: %f" % avg_nlt_eout)
        
        
    
