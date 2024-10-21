import numpy as np

class Line:
    def __init__(self, p1, p2):
        #input: 2 2-dim numpy arrays
        self.p1 = p1
        self.p2 = p2
        diff = np.subtract(p2, p1)
        self.slope = diff[1]/diff[0]
        self.is_vert = False
        #point slope form = y - y1 = m(x - x1) 
        #y = 0 -> -y1 = m(x - x1) -> -y1/m = x - x1 -> (-y1/m) + x1 = x
        self.y_int = ((-1 * p1[1])/self.slope) + p1[0]

        
    def calc(self,testpt):
        #input: numpy array with 2 dim
        
        if self.is_vert == False:
            line_y = self.slope*testpt[0] + self.y_int
            diff = testpt[1] - line_y
        else:
            line_x = self.p1[0]
            diff = testpt[0] - line_x
        return np.sign(diff)

    def calc_pts(self,testpts):
        #testpts should be Nx2
        #goal: test against equation of line, if above, then return +1
        #if on line 0, else -1
        #to check:
        #slope-intercept: y = mx + b or (y-b)/m = x
        if len(testpts.shape) <= 1:
            testpts = testpts.reshape((1,2))
        line_y = np.add(self.y_int, np.multiply(testpts[:,0], self.slope))
        diff = np.subtract(testpts[:,1], line_y)
        return np.sign(diff)
        
        
        
