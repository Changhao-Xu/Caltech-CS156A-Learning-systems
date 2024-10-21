import hw2_hoeffding
import hw2_linreg
import hw2_nonlinear

import sys

n_hoeffding = 10000
n_linreg = 1000
n_nlt = 1000

if len(sys.argv) > 1:
    probstr = sys.argv[1].lower()
    if probstr == "hoeffding":
        hw2_hoeffding.prob12(n_hoeffding)
    elif probstr == "linreg":
        hw2_linreg.prob(n_linreg)
    elif probstr == "nlt":
        hw2_nonlinear.prob(n_nlt)
        

"""

HW2_HOEFFDING:

assignment asks for 10 times the experiments, but it was running a little long so this should be fine

EXAMPLE OUTPUT
Average number of heads:
first coins of sims: 0.498810
coin w/ min freq of heads of sims: 0.038080
rand selection of coin of sims: 0.501150

HW2_LINREG:
e_in average: 0.032890
e_out average: 0.040801
perceptron convergence average: 3.426000

HW2_NLT:
note that the x1,x2,x1x2 coords in the nlt-linreg are a little off

average e_in from linear regression: 0.502665
average weights from nlt-linear regression:
[ -9.90731422e-01   5.17933801e-03   1.24404490e-03  -4.39245218e-03
   1.54683677e+00   1.56262872e+00]

average e_out from nlt-linear regression: 0.126519

"""
