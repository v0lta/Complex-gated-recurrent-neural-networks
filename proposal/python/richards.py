import numpy as np
import matplotlib.pyplot as plt


def step(x):
    return (x > 0).astype(np.float32)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*sigmoid(-x)


def richards(x, k):
    return 1/(1 + np.exp(-x*k))


def richars_prime(x, k):
    return k*richards(x, k)*richards(-x, k)


x = np.linspace(-5, 5, 500)
plt.plot(x, step(x), 'b')
plt.plot(x, sigmoid(x), 'r')
plt.plot(x, sigmoid_prime(x), 'r')
plt.plot(x, richards(x, 5), 'g')
plt.plot(x, richars_prime(x, 5), 'g')
plt.plot(x, richards(x, 10), 'm')
plt.plot(x, richars_prime(x, 10), 'm')
plt.show()
