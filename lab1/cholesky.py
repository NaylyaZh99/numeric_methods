import numpy as np
import scipy.linalg as sla
from math import *
import time
import matplotlib.pyplot as plt

def kholez(n, A):
	for k in range(0, n):
		A[k][k] = sqrt(A[k][k])
		for i in range(k + 1, n):
			A[i][k] = A[i][k] / A[k][k]
		for j in range(k + 1, n):
			for i in range(j, n):
				if j > i:
					A[i][j] = 0
				A[i][j] = A[i][j] - A[i][k] * A[j][k]
	return A

X = np.array(0)
Y = np.array(0)
Y_lib = np.array(0)
n = int(input())
shift = int(input())
wastedTime = 0

while wastedTime <= 1:
	X = np.append(X, n)
	A = np.random.rand(n,n)
	for i in range(n):
		Sum = 0
		for j in range(n):
			if j != i:
				Sum += abs(A[i][j])
		A[i][i] += Sum
	A_lib = np.array(A)
	f = np.random.rand(n)
	f_lib = np.array(f)

	start = time.time()
	A = kholez(n, A)
	print(A)
	wastedTime = time.time() - start
	#print(wastedTime)
	Y = np.append(Y, wastedTime)
	
	start = time.time()
	A_lib = sla.cholesky(A_lib, lower=True)
	print(A_lib)
	wastedTime_lib = time.time() - start
	Y_lib = np.append(Y_lib, wastedTime_lib)
	n = n + shift

print(X)
print(Y)
print(Y_lib)

plt.plot(X, Y)
plt.plot(X, Y_lib)
plt.xlabel('matrix size')
plt.ylabel('sec')
plt.legend(("my realization", "integrated fuction"))
plt.show()
