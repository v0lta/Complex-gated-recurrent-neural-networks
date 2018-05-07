import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def step(x):
    return (x > 0).astype(np.float32)


if 0:
    theta = np.linspace(-np.pi, np.pi, 100)
    a = 1
    b = 2
    gate = sigmoid(np.sin(2*theta + b))
    plt.plot(theta*180/np.pi, gate)
    plt.show()

r = np.linspace(0.1, 100, 500)
a = 1
b = 0.1
gate = np.tanh(1/r*a + b*b)
plt.plot(r, gate)
plt.show()
