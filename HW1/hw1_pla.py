import numpy as np

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
            
        
def Run(trainingpt,runs):
    iterations = []
    ps = []
    for i in range(runs): # calculate convergence iterations and disagreeing probability for each test
        test = Test(trainingpt)
        conv = test.convergence()
        iterations.append(conv)
        p = test.disagreement()
        ps.append(p)
    mean_iterations = np.average(iterations)
    mean_ps = np.average(ps)
    print("N = " + str(trainingpt) + "  " + "iterations = " + str(mean_iterations) )
    print("N = " + str(trainingpt) + "  " + "disagreement P = " + str(mean_ps) )

def main():
    Run(10,10)
    Run(100,10)

if __name__== "__main__":
    main()