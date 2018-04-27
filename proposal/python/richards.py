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
plt.plot(x, richards(x, 1), 'g')
plt.plot(x, richars_prime(x, 1), 'g')
plt.show()

if 0:
    pi = np.pi
    x = np.linspace(-pi, pi, 500)
    for i in range(0, 10, 2):
        for j in range(0, 6, 2):
                plt.plot(richards(np.sin(x + 0.25*j), i+0.001))
    plt.show()
