import numpy as np
import math

class LinearRegression:
    def __init__(self, dim):
        self.weights = np.zeros((dim + 1,1))

    def X_new(self,X):
        examples = X.shape[0]
        X_new = np.c_[np.ones(examples), X]
        return X_new
    
    def train(self,X,Y):
        X_new = self.X_new(X)
        pinv_X = np.linalg.pinv(X_new) # pseudo inverse
        self.weights = np.dot(pinv_X,Y)

    def predict(self,X):
        X_new = self.X_new(X)
        h = np.matmul(X_new, self.weights)
        return h

class NonlinearTrans(LinearRegression):
    def __init__(self, dim, lamda, k):
        self.dim = (2*dim + 3) # 1, x1, x2, x1^2, x2^2, x1*x2, abs(x1-x2), abs(x1+x2)
        self.weights = np.zeros((self.dim + 1, 1))
        self.lamda = lamda
        self.k = k
        
    def change_lambda(self, lamda):
        self.lamda = lamda

    def change_k(self, k):
    	self.k = k

    def X_new(self,X):
        examples = X.shape[0]
        X_multiply = np.prod(X, axis=1) # x1*x2
        X_subtract = np.c_[X[:,0],-X[:,1]] # x1 -x2
        X_new = np.c_[np.ones(examples), X, np.square(X), X_multiply, np.abs(np.sum(X_subtract,axis=1)), np.abs(np.sum(X,axis=1))]
        return X_new[:,:(self.k + 1)]

    def calc_error(self, X,Y):
        examples = X.shape[0]
        predict = np.sign(self.predict(X))
        num_error = np.sum(np.not_equal(predict, np.sign(Y)))
        error = float(num_error)/float(examples)
        return error

    def train_regularization(self, X,Y): # training with regularization: (ZT*Z + lambda*I)^-1 * ZT*y
        X_new = self.X_new(X)
        xTx = np.dot(X_new.T, X_new)
        lI = np.multiply(self.lamda, np.identity(xTx.shape[0])) # lambda*I
        inv_X = np.linalg.inv(np.add(xTx, lI))
        self.weights = np.dot(inv_X, np.dot(X_new.T, Y))

class Import_Data:
    def __init__(self, trainfile, testfile):
        self.dim = 0
        self.train_X, self.train_Y = self.load_file(trainfile)
        self.test_X, self.test_Y = self.load_file(testfile)

    def load_file(self, filename):
        # X = np.array([])
        # Y = np.array([])
        # with open(filename) as file:
        #     data = file.readlines()
        #     self.dim = len(data[0].split()) -1
        #     for line in data:
        #         XY = line.split()
        #         new_XY = [float(k) for k in XY]
        #         X = np.append([X], [new_XY[:-1]], axis = 0)
        #         Y = np.concatenate((X, [new_XY[-1]]))
        # return X, Y
        X = np.array([])
        Y = np.array([])
        with open(filename) as f:
            data = f.readlines()
            examples = len(data)
            self.dim = len(data[0].split()) - 1
            for line in data:
                XY = [float(x) for x in line.split()]
                X = np.concatenate((X, XY[:-1])) #every X but last elt for Y
                Y = np.concatenate((Y, [XY[-1]])) #last elt for Y
        X = X.reshape((examples, self.dim))
        return X, Y

def main():
    data = Import_Data("in.dta", "out.dta")

    lamda = None
    k = 7
    NLT = NonlinearTrans(data.dim, lamda, k) 

    print("25 # raining, 10 # validation")

    for k in np.arange(3,8):
        NLT.change_k(k)
        NLT.train(data.train_X[:25,:], data.train_Y[:25])
        print("k = %d" % k)
        print("Evalidation: %f, Eout: %f" % (NLT.calc_error(data.train_X[25:,:], data.train_Y[25:]), NLT.calc_error(data.test_X, data.test_Y)))

    print("10 # raining, 25 # validation")

    for k in np.arange(3,8):
        NLT.change_k(k)
        NLT.train(data.train_X[25:,:], data.train_Y[25:])
        print("k = %d" % k)
        print("Evalidation: %f, Eout: %f" % (NLT.calc_error(data.train_X[:25,:], data.train_Y[:25]), NLT.calc_error(data.test_X, data.test_Y)))
        
if __name__== "__main__":
    main()