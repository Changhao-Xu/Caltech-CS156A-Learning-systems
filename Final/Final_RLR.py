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

class Import_Data:
    def load_file(self, filename):
        ret_X = np.array([])
        ret_Y = np.array([])
        num_ex = 0 #number of examples
        X_dim = 0 #dimension of data
        with open(filename) as f:
            data = f.readlines()
            num_ex = len(data)
            X_dim = len(data[0].split()) - 1
            for line in data:
                cur_XY = [float(x) for x in line.split()]
                ret_X = np.concatenate((ret_X, cur_XY[1:])) #everything but first elt
                ret_Y = np.concatenate((ret_Y, [cur_XY[0]])) #first elt
        ret_X = ret_X.reshape((num_ex, X_dim))
        self.dim = X_dim
        return ret_X, ret_Y
            
    def __init__(self, trainfile, testfile):
        self.dim = 0
        self.train_X, self.train_Y = self.load_file(trainfile)
        self.test_X, self.test_Y = self.load_file(testfile)
        self.filt_argc = 0

    def Y_mapper(self, Y, eql):
        #maps elts equal to eql to 1 else -1
        return np.subtract(np.multiply(2, np.equal(Y.astype(int), int(eql)).astype(int)), 1)

    def filt_idx(self, Y):
        #returns filtered indices according to my_filt
        return np.where(self.my_filt(Y))[0]

        
    def set_filter(self, params=[]):
        #0 args: no filter
        #1 arg: 1-vs-all - desired digit gets 1, else -1
        #2 args: 1-vs-1 - first digit gets 1, other gets -1, else omitted
        self.filt_argc = min(2, len(params))
        self.filt_argv = params
        if len(params) == 2:
            self.my_filt = np.vectorize(lambda x: int(x) == params[0] or int(x) == params[1])

    def get_X(self, req_set = "train"):
        if req_set.lower() == "train".lower():
            if self.filt_argc == 0 or self.filt_argc == 1:
                return self.train_X
            elif self.filt_argc == 2:
                #filtered indices
                filtered = self.filt_idx(self.train_Y)
                return self.train_X[filtered]
        else:
            if self.filt_argc == 0 or self.filt_argc == 1:
                return self.test_X
            elif self.filt_argc == 2:
                #filtered indices
                filtered = self.filt_idx(self.test_Y)
                return self.test_X[filtered]

    def get_Y(self, req_set = "train"):
        if req_set.lower() == "train".lower():
            if self.filt_argc == 0:
                return self.train_Y
            elif self.filt_argc == 1:
                #one-liner for mapping given param as 1 else -1
                return self.Y_mapper(self.train_Y, self.filt_argv[0])
            elif self.filt_argc == 2:
                #filtered indices
                filtered = self.filt_idx(self.train_Y)
                return self.Y_mapper(self.train_Y[filtered], self.filt_argv[0])
        else:
            if self.filt_argc == 0:
                return self.test_Y
            elif self.filt_argc == 1:
                #one-liner for mapping given param as 1 else -1
                return self.Y_mapper(self.test_Y, self.filt_argv[0])
            elif self.filt_argc == 2:
                #filtered indices
                filtered = self.filt_idx(self.test_Y)
                return self.Y_mapper(self.test_Y[filtered], self.filt_argv[0])

def main():
	data = Import_Data("features.train", "features.test")

	print("Q7~Q9 are based on the following results:")

	RegLR = RegularizedLinearRegression(data.dim,1) # lambda = 1
	for cur_num in range(10):
	    data.set_filter([cur_num]) #setting to x-vs-all
	    X_train = data.get_X("train")
	    Y_train= data.get_Y("train")
	    X_test = data.get_X("test")
	    Y_test= data.get_Y("test")
	    RegLR.train_regularization(X_train, Y_train)
	    print("%d versus all w/o NLT: Ein = %f, Eout = %f" % (cur_num, RegLR.calc_error(X_train, Y_train), RegLR.calc_error(X_test, Y_test)))

	print("")
	
	NLT = NonlinearTrans(data.dim,1) # lambda = 1
	for cur_num in range(10):
	    data.set_filter([cur_num]) #setting to x-vs-all
	    X_train = data.get_X("train")
	    Y_train= data.get_Y("train")
	    X_test = data.get_X("test")
	    Y_test= data.get_Y("test")
	    NLT.train_regularization(X_train, Y_train)
	    print("%d versus all w/ NLT: Ein = %f, Eout = %f" % (cur_num, NLT.calc_error(X_train, Y_train), NLT.calc_error(X_test, Y_test)))

	print("")
	print("Q10 is based on the following results:")
	
	data.set_filter([1, 5]) #setting to 1-vs-5
	X_train = data.get_X("train")
	Y_train= data.get_Y("train")
	X_test = data.get_X("test")
	Y_test= data.get_Y("test")
	
	print("When lambda = 0.01,")
	NLT.change_lambda(0.01) # lambda = 0.01
	NLT.train_regularization(X_train, Y_train)
	print("Ein = %f, Eout = %f" % (NLT.calc_error(X_train, Y_train), NLT.calc_error(X_test, Y_test)))
	
	print("When lambda = 1,")
	NLT.change_lambda(1) # lambda = 0.01
	NLT.train_regularization(X_train, Y_train)
	print("Ein = %f, Eout = %f" % (NLT.calc_error(X_train, Y_train), NLT.calc_error(X_test, Y_test)))

if __name__== "__main__":
    main()