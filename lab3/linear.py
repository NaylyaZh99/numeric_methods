import numpy as np
import matplotlib.pyplot as plt

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
	for j in range(n):
		if x[j] <= xnew[i] <= x[j + 1]:
			k = (y[j + 1] - y[j]) / (x[j + 1] - x[j])
			ynew[i] = k * (xnew[i] - x[j]) + y[j]
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


    

