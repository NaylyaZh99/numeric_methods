import numpy as np
import time
import matplotlib.pyplot as plt

def gauss(n, A, f):
	res = np.zeros(n)
	for k in range(n):
		for j in range(k + 1, n):
			A[k][j] = A[k][j] / A[k][k]
		f[k] = f[k] / A[k][k]
		A[k][k] = 1
		for i in range(k + 1, n):
			for j in range(k + 1, n):
				A[i][j] = A[i][j] - A[k][j] * A[i][k];
			f[i] = f[i] - f[k] * A[i][k]
			A[i][k] = 0
	for i in range(n - 1, -1, -1):
		res[i] = f[i]
		for j in range(i + 1, n):
			res[i] = res[i] - A[i][j] * res[j]
	return res

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
	x = gauss(n, A, f)
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
