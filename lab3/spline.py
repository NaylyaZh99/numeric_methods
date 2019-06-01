import numpy as np
import matplotlib.pyplot as plt

def sweep(a, b, c, f, n):
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


def generateSpline(x, y):
    n = len(x) - 1
    h = (x[n] - x[0]) / n
    
    a = np.array([0] + [1] * (n - 1) + [0])
    b = np.array([1] + [4] * (n - 1) + [1])
    c = np.array([0] + [1] * (n - 1) + [0])
    f = np.array([0] * (n + 1))
    for i in range(1, n):
        f[i] = 3 * (y[i - 1] - 2 * y[i] + y[i + 1]) / h ** 2
    s = sweep(a, b, c, f, n + 1)
    A = np.array([0.0] * (n + 1))
    B = np.array([0.0] * (n + 1))
    C = np.array([0.0] * (n + 1))
    D = np.array([0.0] * (n + 1))
    for i in range(n):
        B[i] = s[i]
        A[i] = (B[i + 1] - B[i]) / (3 * h)
        C[i] = (y[i + 1] - y[i]) / h - (B[i + 1] + 2 * B[i]) * h / 3
        D[i] = y[i]
    return A, B, C, D

X = open('train.dat', 'r')
Y = open('train.ans', 'r')
Z = open('test.dat', 'r')
F = open('test.ans', 'w')

x = [float(i) for i in X.readline().split()]
y = [float(i) for i in Y.readline().split()]
xnew = [float(i) for i in Z.readline().split()]

n = len(x)
m = len(xnew)

ynew = np.zeros(m)

A, B, C, D = generateSpline(x, y)

for i in range(m):
	for j in range(n):
		if x[j] <= xnew[i] <= x[j + 1]:
			ynew[i] = A[j] * ((xnew[i] - x[j]) ** 3) + B[j] * ((xnew[i] - x[j]) ** 2) + C[j] * (xnew[i] - x[j]) + D[j]
			F.write(str(ynew[i]) + ' ')
			break
X.close()
Y.close()
Z.close()
F.close()

plt.plot(x, y)
plt.plot(xnew, ynew)
plt.scatter(x, y, s=10, c='blue')
plt.scatter(xnew, ynew, s=10, c='orange')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(("given", "interpolation"))
plt.grid()
plt.show()
