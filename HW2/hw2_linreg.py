import numpy as np

class Regression:
    def __init__(self, dim):
        self.dim = dim
        self.weights = np.zeros(1 + dim) # need one more dim for threshold

    def X_reshape(self,X):
        num_examples = X.shape[0]
        real_X = np.c_[np.ones(num_examples), X] # add one more column for threshold
        return real_X
    
    def predict(self,X):
        real_X = self.X_reshape(X)
        cur_h = np.matmul(real_X, self.weights) # real x is (n*(d+1)) matrix, while weights is ((d+1)*1) matrix
        return cur_h

    def train(self,X,Y):
        #for the sake of programming ease, let's just assume inputs are numpy ndarrays
        #and are the proper shapes (X = (n, dim), y = (n,1))
        real_X = self.X_reshape(X)
        pinv_X = np.linalg.pinv(real_X) # compute pseudo-inverse of X
        self.weights = np.dot(pinv_X,Y) # w = pseudo-inverse * Y

class Perceptron:
    def __init__(self, dim):
        self.dim = dim
        self.weights = np.zeros(1 + dim) # need one more dim for threshold

    def predict(self,x):
        newx = np.append([1],x[:self.dim]) # add artificial coordinate x0 = 1
        current_h = np.dot(self.weights, newx) # current hypothese set h
        return np.sign(current_h)
    
    def train(self,x,y):
        predict = self.predict(x)
        if predict == y:
            return 1
        else: # misclassified training points
            newx = np.append([1],x[:self.dim])
            self.weights = self.weights + np.multiply(y,newx) # w = w + y * xn
        return 0

class Vector:
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

class Test:        
    def __init__(self, trainingpt):
        self.n = trainingpt
        self.points = [np.random.uniform(-1,1,2) for x in range(trainingpt)] # generate 'trainingpt' of random points
        p = [np.random.uniform(-1,1,2) for x in range(2)]
        while p[0][0] == p[1][0] and p[0][1] == p[1][1]:
            p = [np.random.uniform(-1,1,2) for x in range(2)] # pick up two different points
        self.target = Vector(p[0],p[1]) # define target function f
        self.perceptron = Perceptron(2)

    def agree(self,point):
        current_predict = self.perceptron.predict(point) 
        actual = self.target.compare(point) # target function output
        return current_predict == actual # compare current predict with target output

    def disagreement(self):
        n = 0
        for x in range(1000): # map 1000 points to calculate disagreeing probability
            agree = self.agree(np.random.uniform(-1,1,2))
            if not agree:
                n = n + 1
        prob = float(n)/1000.0
        return prob

    def convergence(self):
        iterations = 0
        testpt = 0
        while True:
            actual = self.target.compare(self.points[testpt])
            success = self.perceptron.train(self.points[testpt],actual)
            if success:
                converge = True # training finished for current iteration, assume convergence for now
                for i in range(self.n):
                    agree = self.agree(self.points[i])
                    if not agree:
                        converge = False
                        break
                if converge:
                    break
            else:
                iterations = iterations + 1
            testpt = int(np.random.uniform(0, self.n-0.1)) # pick up another test point
        return iterations


#E_in(w) = (1/N)*L2norm(X*w-y)
class Rtest:        
    def __init__(self, numpoints):
        self.n = numpoints
        self.points = np.random.uniform(-1.0,1.0,(self.n, 2))
        p = [np.random.uniform(-1.0,1.0,2) for x in range(2)]
        while p[0][0] == p[1][0] and p[0][1] == p[1][1]:
            p = [np.random.uniform(-1.0,1.0,2) for x in range(2)]
        self.target = Vector(p[0],p[1])
        self.labels = np.array([self.target.compare(x) for x in self.points])
        self.lr = Regression(2)

    def regen_points(self, numpoints):
        self.n = numpoints
        self.points = np.random.uniform(-1.0,1.0,(self.n, 2))
        self.labels = np.array([self.target.compare(x) for x in self.points])

    def train(self):
        self.lr.train(self.points, self.labels)
        
    def e_in(self):
        xw = self.lr.predict(self.points)
        xw = np.sign(xw)
        mydiff = np.not_equal(xw, self.labels)
        e_in = np.mean(mydiff)
        #print(e_in)
        #e_in = np.multiply(1.0/float(self.n), mydiff)
        return e_in
        
def prob(num_exp):
    n1 = 100
    n2 = 1000
    n_pla = 10
    ein = np.array([])
    eout = np.array([])
    iters = np.array([])
    for i in range(num_exp):
        cur_lr = Rtest(n1)
        cur_lr.train()
        cur_ein = cur_lr.e_in()
        ein = np.concatenate((ein,[cur_ein]))
        cur_lr.regen_points(n2)
        cur_eout = cur_lr.e_in()
        eout = np.concatenate((eout,[cur_eout]))
        cur_pla = Test(n_pla)
        cur_pla.target = cur_lr.target
        cur_pla.perceptron.weights = cur_lr.lr.weights.T
        cur_iter = cur_pla.convergence()
        iters = np.concatenate((iters,[cur_iter]))
    ein_avg = np.average(ein)
    eout_avg = np.average(eout)
    iters_avg = np.average(iters)
    print("e_in average: %f" % ein_avg)
    print("e_out average: %f" % eout_avg)
    print("perceptron convergence average: %f" % iters_avg)


def main():
    prob(1000)

if __name__== "__main__":
    main()
