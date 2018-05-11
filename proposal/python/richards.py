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

if 0:
    x = np.linspace(-5, 5, 500)
    plt.plot(x, richards(x, 4), 'g')
    plt.plot(x, richars_prime(x, 4), 'g')
    plt.show()

if 1:
    pi = np.pi
    x = np.linspace(-pi, pi, 500)
    a = np.linspace(-5.0, 5.0, 4)  # open closed bias
    b = np.linspace(-pi, pi, 4)  # rotates the filter.
    c = np.linspace(0, 5, 4)  # changes steepness positive
    d = np.linspace(0, 2, 4)
    # plt.plot(x, sigmoid(x))
    # plt.show()
    for i in range(0, a.shape[0]):
        for j in range(0, b.shape[0]):
            for k in range(0, c.shape[0]):
                plt.plot(x, sigmoid(c[k]*np.sin(x + b[j]) + a[i]))
                # for l in range(0, d.shape[0]):
                #    plt.plot(x, sigmoid(np.sin(x*d[l])))
                #    # plt.plot(x, richards(, i+0.001))
    plt.show()
