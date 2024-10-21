import numpy as np
# Open a file
fo = open("in.dta")
print ("Name of the file: ", fo.name)
data = fo.readlines()
print (data[0].split())
# [float(x) for x in data[0].split()]
k = np.array([])
# print(np.concatenate((k, [2])))
for x in data[0].split():
	k = np.concatenate((k, [float(x)]))
print (k)