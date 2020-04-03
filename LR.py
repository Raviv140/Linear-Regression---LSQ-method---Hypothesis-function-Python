import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

N = 200
x = sy.Symbol('x')

xn = np.random.uniform(1, 10, N)
yn = np.random.uniform(3, 15, N)
xx = np.linspace(0, 15, N)

xx = xx + xn
y = 1.74 * xx + 4.5 + yn
plt.title(f"Linear Regression \nFinding the best linear line - Hypothesis function\n"
          f"{chr(952)}(x) = {chr(952)}1x + {chr(952)}0", color='r')
plt.axis([-2, 30, -2, 90])
plt.plot(xx, y, '.m')
plt.pause(2)

A = np.zeros((2, 2), dtype=np.float64)
A[0, 0] = np.sum(xx ** 2)
A[0, 1] = A[1, 0] = np.sum(xx)
A[1, 1] = N
b = np.zeros((2, 1), dtype=np.float64)
b[0, :] = np.sum(xx * y)
b[1, :] = np.sum(y)
hyp = np.dot(np.linalg.pinv(A), b)
hypothesis = round(hyp[0, 0], 3)*x+round(hyp[1, 0], 3)
print(f"The optimal linear line is : \n f(x) = {hypothesis}")
y_hypothesis = hyp[0, 0] * xx + hyp[1, 0]
plt.text(15, 15, f"The optimal linear line :\n f(x) = {hypothesis}\n"
                 f"{chr(952)}1 ~ {round(hyp[0, 0], 3)}\n"
                 f"{chr(952)}0 ~ {round(hyp[1, 0], 3)}")
plt.plot(xx, y_hypothesis, 'orange', linewidth=2)
plt.show()
