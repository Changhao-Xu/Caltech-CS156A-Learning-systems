import numpy as np
from line_class import Line

#generates target function (random line), random points, and labels
class LineTest:
    def __init__(self, num_train, num_test):
        self.thresh = 0.001
        line_pts = np.zeros((2,2))
        #don't want line vertical or near vertical
        while abs(line_pts[0][1] - line_pts[1][1]) <= self.thresh:
            line_pts = np.random.uniform(-1,1, (2,2))
        self.line = Line(line_pts[0], line_pts[1])
        num_train = max(1, num_train)
        self.num_train = num_train
        self.train_set = np.random.uniform(-1,1, (num_train, 2))
        self.train_labels = self.line.calc_pts(self.train_set)
        num_test = max(1,num_test)
        self.num_test = num_test
        self.test_set = np.random.uniform(-1,1, (num_test, 2))
        self.test_labels = self.line.calc_pts(self.test_set)
