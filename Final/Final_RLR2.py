import numpy as np

class RegularizedLinearRegression:
    def __init__(self, dim, lamda):
        self.weights = np.zeros((dim + 1,1))
        self.lamda = lamda

    def X_new(self,X):
        examples = X.shape[0]
        X_new = np.c_[np.ones(examples), X]
        return X_new
    
    def train_regularization(self, X,Y): # training with regularization: (ZT*Z + lambda*I)^-1 * ZT*y
        X_new = self.X_new(X)
        xTx = np.dot(X_new.T, X_new)
        lI = np.multiply(self.lamda, np.identity(xTx.shape[0])) # lambda*I
        inv_X = np.linalg.inv(np.add(xTx, lI))
        self.weights = np.dot(inv_X, np.dot(X_new.T, Y))

    def predict(self,X):
        X_new = self.X_new(X)
        h = np.matmul(X_new, self.weights)
        return h

    def calc_error(self, X,Y):
        examples = X.shape[0]
        predict = np.sign(self.predict(X))
        num_error = np.sum(np.not_equal(predict, np.sign(Y)))
        error = float(num_error)/float(examples)
        return error

class NonlinearTrans(RegularizedLinearRegression):
    def __init__(self, dim, lamda):
        self.dim = (2*dim + 1) # 1, x1, x2, x1^2, x2^2, x1*x2
        self.weights = np.zeros((self.dim + 1, 1))
        self.lamda = lamda
        
    def change_lambda(self, lamda):
        self.lamda = lamda

    def X_new(self,X):
        examples = X.shape[0]
        X_multiply = np.prod(X, axis=1) # x1*x2
        X_subtract = np.c_[X[:,0],-X[:,1]] # x1 -x2
        X_new = np.c_[np.ones(examples), X, X_multiply, np.square(X)]
        return X_new

def set_labels_all(x, Y): # x = desired digit in 'x versus all'
    Y_labeled = np.array([])
    for i in np.arange(Y.shape[0]):
        if Y[i] == x:
            Y_labeled = np.append(Y_labeled, 1)
        else:
            Y_labeled = np.append(Y_labeled, -1)
    return Y_labeled

def set_labels_one(x1, x2, X, Y): # x1, x2 = desired digits in 'x1 versus x2'
    X_labeled = np.array([])
    Y_labeled = np.array([])
    for i in np.arange(Y.shape[0]):
        if Y[i] == x1:
            X_labeled = np.concatenate((X_labeled, X[i,:]))
            Y_labeled = np.append(Y_labeled, 1)
        if Y[i] == x2:
            X_labeled = np.concatenate((X_labeled, X[i,:]))
            Y_labeled = np.append(Y_labeled, -1)
    X_labeled = X_labeled.reshape((Y_labeled.shape[0], X.shape[1]))
    return X_labeled, Y_labeled

def load_data(filename_train, filename_test):
    data_train = np.loadtxt(filename_train)
    X_train = data_train[:, 1:] # delete 1st digit term (Y)
    Y_train = data_train[:, :1] # extract 1st digit term (Y)
    data_test = np.loadtxt(filename_test)
    X_test = data_test[:, 1:] # delete 1st digit term (Y)
    Y_test = data_test[:, :1] # extract 1st digit term (Y)
    # data = np.r_[data_train, data_test]
    return X_train, Y_train, X_test, Y_test

def main():
    X_train, Y_train, X_test, Y_test = load_data('features.train','features.test')
    
    print("Q7~Q9 are based on the following results:")

    RegLR = RegularizedLinearRegression(X_train.shape[1], 1) # lambda = 1
    for x in range(10):
        Y_train_all = set_labels_all(x, Y_train)
        Y_test_all = set_labels_all(x, Y_test)
        RegLR.train_regularization(X_train, Y_train_all)
        print("%d versus all w/o NLT: Ein = %f, Eout = %f" % (x, RegLR.calc_error(X_train, Y_train_all), RegLR.calc_error(X_test, Y_test_all)))

    print("")
    
    NLT = NonlinearTrans(X_train.shape[1], 1) # lambda = 1
    for x in range(10):
        Y_train_all = set_labels_all(x, Y_train)
        Y_test_all = set_labels_all(x, Y_test)
        NLT.train_regularization(X_train, Y_train_all)
        print("%d versus all w/ NLT: Ein = %f, Eout = %f" % (x, NLT.calc_error(X_train, Y_train_all), NLT.calc_error(X_test, Y_test_all)))

    print("")
    print("Q10 is based on the following results:")
    
    X_train_one, Y_train_one = set_labels_one(1, 5, X_train, Y_train)
    X_test_one, Y_test_one = set_labels_one(1, 5, X_test, Y_test)
    
    print("When lambda = 0.01,")
    NLT.change_lambda(0.01) # lambda = 0.01
    NLT.train_regularization(X_train_one, Y_train_one)
    print("Ein = %f, Eout = %f" % (NLT.calc_error(X_train_one, Y_train_one), NLT.calc_error(X_test_one, Y_test_one)))
    
    print("When lambda = 1,")
    NLT.change_lambda(1) # lambda = 0.01
    NLT.train_regularization(X_train_one, Y_train_one)
    print("Ein = %f, Eout = %f" % (NLT.calc_error(X_train_one, Y_train_one), NLT.calc_error(X_test_one, Y_test_one)))

if __name__== "__main__":
    main()