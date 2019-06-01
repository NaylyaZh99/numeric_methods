import numpy as np
from math import *
import time 
import matplotlib.pyplot as plt

def seidel(n, A, f, x):
	res = np.zeros(n)
	for i in range(n):
		s = 0
		for j in range(i):
			s = s + A[i][j] * res[j]
		for j in range(i+1, n):
			s = s + A[i][j] * x[j]
		res[i] = (f[i] - s) / A[i][i]
	return res

def diff(n, x, y):
	s = 0
	for i in range(n):
		s += (x[i] - y[i]) ** 2
	return sqrt(s)

def solve(n, A, f):
	res = np.zeros(n)
	while True:
		x = np.array(res)
		res = seidel(n, A, f, x)
		if diff(n, x, res) < 0.005:
			break
	return res

X = np.array(0)
Y = np.array(0)
Y_lib = np.array(0)
n = int(input())
shift = int(input())
wastedTime = 0

while wastedTime <= 0.001:
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
	x = solve(n, A, f)
	wastedTime = time.time() - start
	print(wastedTime)
	Y = np.append(Y, wastedTime)
	
	start = time.time()
	x_lib = np.linalg.solve(A_lib, f_lib)
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

