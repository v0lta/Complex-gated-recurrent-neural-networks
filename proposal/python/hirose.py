import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(-8, 8, 500)
# CRE
if 1:
    plt.plot(r, np.power(1/np.cosh(r), 2))
    plt.plot(r, np.tanh(r)/r)
    plt.plot(r, np.abs(np.power(1/np.cosh(r), 2) - np.tanh(r)/(r)))
    plt.show()

# z derivatives.
# plt.plot(r, np.abs(np.power(1/np.cosh(r), 2) - np.tanh(r)/(r)))
# plt.show()
