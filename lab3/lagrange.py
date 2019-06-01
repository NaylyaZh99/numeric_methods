import numpy as np
import matplotlib.pyplot as plt

def phi(i, z):
	p = 1.0
	for j in range(n):
		if i != j:
			p = p * (z - x[j]) / (x[i] - x[j])
	return p

def Lagrange(x, y, z):
	s = 0.0
	for i in range(n):
		if z == x[i]:
			return y[i]
	for i in range(n):
		s = s + y[i] * phi (i, z)   
	return s

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

for i in range(m):
    ynew[i] = Lagrange(x, y, xnew[i])
    F.write(str(ynew[i]) + ' ')


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


    

