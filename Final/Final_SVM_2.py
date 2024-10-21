import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt

X_train = np.array([[1,0],[0,1], [0,-1], [-1,0], [0,2],[0,-2], [-2,0]])
Y_train = np.array([-1,-1,-1,1,1,1,1])

x1 = X_train[:, :1]
x2 = X_train[:, 1:]

z1 = pow(x2,2.0) - 2 * x1 - 1
z2 = pow(x1,2.0) - 2 * x2 + 1

z1p = np.array([z1[i] for i in np.where(Y_train == 1)]).flatten()
z2p = np.array([z2[i] for i in np.where(Y_train == 1)]).flatten()
z1n = np.array([z1[i] for i in np.where(Y_train == -1)]).flatten()
z2n = np.array([z2[i] for i in np.where(Y_train == -1)]).flatten()

plt.plot(z1p, z2p, 'r+', label='+1')
plt.plot(z1n, z2n, 'bx', label='-1')
plt.plot([0.5, 0.5], [-3.0, 5.0], 'g-', label='seperating plane')
plt.xlabel('z1')
plt.ylabel('z2')
plt.legend()
plt.show()

clf = svm.SVC(C = np.inf, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1) # C = infinite for hard-margin SVM with 2nd order polynomial kernel with intercept coef0 = 1
# For simplicity, take gamma = 1, same in Lecture 15 page 7
Z = np.c_[z1, z2]
clf.fit(Z, Y_train)
print("number of support vectors: ", sum(clf.n_support_))