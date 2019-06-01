import numpy as np
import scipy.linalg as sla
import time
import matplotlib.pyplot as plt

def sweep(n, a, b, c, f):
	alpha = np.zeros(n + 1)
	beta = np.zeros(n + 1)
	x = np.zeros(n)
	
	for i in range(n):
		d = a[i] * alpha[i] + b[i]
		alpha[i + 1] = -c[i] / d
		beta[i + 1] = (f[i] - a[i] * beta[i]) / d
	x[n - 1] = beta[n]
	for i in range(n - 2, -1, -1):
		x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]
	return x

X = np.array(0)
Y = np.array(0)
Y_lib = np.array(0)
n = int(input())
shift = int(input())
wastedTime = 0

while wastedTime <= 1:
	A = np.zeros((3, n))
	a = np.random.rand(n)
	b = np.random.rand(n)
	c = np.random.rand(n)
	f = np.random.rand(n)
	a[0], c[n - 1] = 0, 0
	for i in range(n):
		b[i] = abs(a[i]) + abs(b[i]) + abs(c[i]) + 1
		A[1][i] = b[i]
		if i > 0:
			A[2][i] = c[i - 1]
		if i < n - 1:
			A[0][i] = a[i + 1]
	A_lib = np.array(A)
	f = np.random.rand(n)
	f_lib = np.array(f)
	X = np.append(X, n)

	start = time.time()
	x = sweep(n, a, b, c, f)
	print(x)
	wastedTime = time.time() - start
	print(wastedTime)
	Y = np.append(Y, wastedTime)
	
	start = time.time()
	x_lib = sla.solve_banded((1, 1), A_lib, f_lib)
	print(x_lib)
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

