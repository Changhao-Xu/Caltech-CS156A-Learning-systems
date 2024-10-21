import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt

# X_train = np.array([[1,0],[0,1], [0,-1], [-1,0], [0,2],[0,-2], [-2,0]])
# X_train = np.array([-1,-1,-1,1,1,1,1])

RawData = [[1, 0, -1], [0, 1, -1], [0, -1, -1], [-1, 0, 1], [0, 2, 1], [0, -2, 1], [-2, 0, 1]]
data = pd.DataFrame(RawData, columns = ['x1', 'x2', 'y'], dtype=np.float64)

x1 = data['x1']
x2 = data['x2']
y = data['y']

z1 = pow(x2,2.0) - 2 * x1 - 1
z2 = pow(x1,2.0) - 2 * x2 + 1

plt.plot(z1[y==1], z2[y==1], 'r+', label='+1')
plt.plot(z1[y==-1], z2[y==-1], 'bx', label='-1')
plt.plot([0.5, 0.5], [-3.0, 5.0], 'g-', label='seperating plane')
plt.xlabel('z1')
plt.ylabel('z2')
plt.legend()
plt.show()

clf = svm.SVC(C = np.inf, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1) # C = infinite for hard-margin SVM with 2nd order polynomial kernel with intercept coef0 = 1
# For simplicity, take gamma = 1, same in Lecture 15 page 7
Z = np.c_[z1, z2]
clf.fit(Z, y)
print("number of support vectors: ", sum(clf.n_support_))