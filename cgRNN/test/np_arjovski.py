import numpy as np
import matplotlib.pyplot as plt

shape = [128, 128]
assert shape[0] == shape[1]
omega1 = np.random.uniform(-np.pi, np.pi, shape[0])
omega2 = np.random.uniform(-np.pi, np.pi, shape[0])
omega3 = np.random.uniform(-np.pi, np.pi, shape[0])

vr1 = np.random.uniform(-1, 1, [shape[0], 1])
vi1 = np.random.uniform(-1, 1, [shape[0], 1])
v1 = vr1 + 1j*vi1
vr2 = np.random.uniform(-1, 1, [shape[0], 1])
vi2 = np.random.uniform(-1, 1, [shape[0], 1])
v2 = vr2 + 1j*vi2

D1 = np.diag(np.exp(1j*omega1))
D2 = np.diag(np.exp(1j*omega2))
D3 = np.diag(np.exp(1j*omega3))

vvh1 = np.matmul(v1, np.transpose(np.conj(v1)))
beta1 = 2./np.matmul(np.transpose(np.conj(v1)), v1)
R1 = np.eye(shape[0]) - beta1*vvh1

vvh2 = np.matmul(v2, np.transpose(np.conj(v2)))
beta2 = 2./np.matmul(np.transpose(np.conj(v2)), v2)
R2 = np.eye(shape[0]) - beta2*vvh2

perm = np.random.permutation(np.eye(shape[0], dtype=np.float32)) \
    + 1j*np.zeros(shape[0])

fft = np.fft.fft
ifft = np.fft.ifft

step1 = fft(D1)
step2 = np.matmul(R1, step1)
step3 = np.matmul(perm, step2)
step4 = np.matmul(D2, step3)
step5 = ifft(step4)
step6 = np.matmul(R2, step5)
unitary = np.matmul(D3, step6)
eye_test = np.matmul(np.transpose(np.conj(unitary)), unitary)
unitary_test = np.linalg.norm(np.eye(shape[0]) - eye_test)
print('result: ', unitary_test)
