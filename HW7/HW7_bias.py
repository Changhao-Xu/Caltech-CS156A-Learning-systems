import numpy as np

N = 100000000

e1 = np.random.uniform(0,1,N)
e2 = np.random.uniform(0,1,N)
min_e1_e2 = np.minimum(e1, e2)

print(np.mean(e1))
print(np.mean(e2))
print(np.mean(min_e1_e2))