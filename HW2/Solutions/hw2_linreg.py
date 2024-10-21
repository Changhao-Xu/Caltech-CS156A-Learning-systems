import linreg
import numpy as np
import hw1_pla

class Line:
    def __init__(self, p1, p2):
        #input: 2 2-dim numpy arrays
        self.p1 = p1
        self.p2 = p2
        diff = np.subtract(p2, p1)
        if diff[0] <= 0.001:
            #if vertical (or close to it), just set slope to none
            self.slope = None
            self.is_vert = True
        else:
            self.slope = diff[1]/diff[0]
            self.is_vert = False
        #point slope form = y - y1 = m(x - x1) 
        #y = 0 -> -y1 = m(x - x1) -> -y1/m = x - x1 -> (-y1/m) + x1 = x
        if not self.is_vert:
            self.y_int = ((-1 * p1[1])/self.slope) + p1[0]

        
    def calc(self,testpt):
        #input: numpy array with 2 dim
        #goal: test against equation of line, if above, then return +1
        #if on line 0, else -1
        #to check:
        #if vertical, check against x, else check against y

        #slope-intercept: y = mx + b or (y-b)/m = x
        if self.is_vert == False:
            line_y = self.slope*testpt[0] + self.y_int
            diff = testpt[1] - line_y
        else:
            line_x = self.p1[0]
            diff = testpt[0] - line_x
        return np.sign(diff)

#E_in(w) = (1/N)*L2norm(X*w-y)
class LRtest:        
    def __init__(self, numpoints):
        self.n = numpoints
        self.points = np.random.uniform(-1.0,1.0,(self.n, 2))
        p = [np.random.uniform(-1.0,1.0,2) for x in range(2)]
        while p[0][0] == p[1][0] and p[0][1] == p[1][1]:
            p = [np.random.uniform(-1.0,1.0,2) for x in range(2)]
        self.target = Line(p[0],p[1])
        self.labels = np.array([self.target.calc(x) for x in self.points])
        self.lr = linreg.LinReg(2)

    def regen_points(self, numpoints):
        self.n = numpoints
        self.points = np.random.uniform(-1.0,1.0,(self.n, 2))
        self.labels = np.array([self.target.calc(x) for x in self.points])

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
        cur_lr = LRtest(n1)
        cur_lr.train()
        cur_ein = cur_lr.e_in()
        ein = np.concatenate((ein,[cur_ein]))
        cur_lr.regen_points(n2)
        cur_eout = cur_lr.e_in()
        eout = np.concatenate((eout,[cur_eout]))
        cur_pla = hw1_pla.PLAtest(n_pla)
        cur_pla.target = cur_lr.target
        cur_pla.PLA.weights = cur_lr.lr.weights.T
        cur_iter = cur_pla.test_convergence()
        iters = np.concatenate((iters,[cur_iter]))
    ein_avg = np.average(ein)
    eout_avg = np.average(eout)
    iters_avg = np.average(iters)
    print("e_in average: %f" % ein_avg)
    print("e_out average: %f" % eout_avg)
    print("perceptron convergence average: %f" % iters_avg)
