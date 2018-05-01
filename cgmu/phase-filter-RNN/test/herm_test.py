import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

shape = [512, 512]
limit = np.sqrt(6 / (shape[0] + shape[1]))
rand_r = np.random.uniform(-limit, limit, shape[0:2])
rand_i = np.random.uniform(-limit, limit, shape[0:2])
crand = rand_r + 1j*rand_i
u, s, vh = np.linalg.svd(crand)
# use u and vg to create a unitary matrix:
unitary = np.matmul(u, np.transpose(np.conj(vh)))

rand_gr = np.random.uniform(-limit, limit, shape[0:2])
rand_gi = np.random.uniform(-limit, limit, shape[0:2])
W = unitary
test0 = np.eye(512) - np.matmul(np.transpose(np.conj(W)), W)
print('unitary W ', np.linalg.norm(test0))

G = rand_gr + 1j*rand_gi
# set up skew-symmetric matrix A.
A = np.matmul(np.transpose(np.conj(G)), W) - np.matmul(np.transpose(np.conj(W)), G)

test = np.transpose(np.conj(A)) - (-A)
print('skew    A ', np.linalg.norm(test))
# test2 = np.matmul(np.transpose(np.conj(A)), A)
# print('orthogonal A', np.linalg.norm(test2))

lr = 0.01
denom = np.eye(512) + (0.01/2.0 * A)
numer = np.eye(512) - (0.01/2.0 * A)
caley = np.matmul(np.linalg.inv(denom), numer)
test3 = np.eye(512) - np.matmul(np.transpose(np.conj(caley)), caley)
print('unitary C ', np.linalg.norm(test3))

w_new = np.matmul(caley, W)
test4 = np.eye(512) - np.matmul(np.transpose(np.conj(w_new)), w_new)
print('unitary Wn', np.linalg.norm(test4))


# --------------------------- Tensorflow --------------
def unitary_init(shape, dtype=np.float32, partition_info=None):
    limit = np.sqrt(6 / (shape[0] + shape[1]))
    rand_r = np.random.uniform(-limit, limit, shape[0:2])
    rand_i = np.random.uniform(-limit, limit, shape[0:2])
    crand = rand_r + 1j*rand_i
    u, s, vh = np.linalg.svd(crand)
    # use u and vg to create a unitary matrix:
    unitary = np.matmul(u, np.transpose(np.conj(vh)))
    print('unitary norm:', np.linalg.norm(unitary))
    # test
    # plt.imshow(np.abs(np.matmul(unitary, np.transpose(np.conj(unitary))))); plt.show()
    stacked = np.stack([np.real(unitary), np.imag(unitary)], -1)
    assert stacked.shape == tuple(shape), "Unitary initialization shape mismatch."
    # debug_here()
    return tf.constant(stacked, dtype)


test_graph = tf.Graph()
with test_graph.as_default():
    dtype = tf.float64
    cdtype = tf.complex128
    var = tf.get_variable('W', shape=[512, 512, 2], initializer=unitary_init,
                          dtype=dtype)
    W = tf.complex(var[:, :, 0], var[:, :, 1])
    eye = tf.eye(*tf.Tensor.get_shape(W).as_list(), dtype=W.dtype)
    test_w = eye - tf.matmul(tf.transpose(tf.conj(W)), W)
    test_w_norm = tf.norm(test_w)
    G = tf.complex(tf.random_uniform([512, 512], dtype=dtype),
                   tf.random_uniform([512, 512], dtype=dtype))
    grad_shape = tf.Tensor.get_shape(G).as_list()
    assert grad_shape[0] == grad_shape[1]
    eye = tf.eye(grad_shape[0], dtype=cdtype)
    A = tf.matmul(tf.conj(tf.transpose(G)), W) \
        - tf.matmul(tf.conj(tf.transpose(W)), G)
    test_a = tf.transpose(tf.conj(A)) - (-A)
    test_a_norm = tf.norm(test_a)
    cayleyDenom = eye + 0.001/0.2 * A
    cayleyNumer = eye - 0.001/0.2 * A
    C = tf.matmul(tf.matrix_inverse(cayleyDenom), cayleyNumer)
    eye = tf.eye(*tf.Tensor.get_shape(C).as_list(), dtype=C.dtype)
    test_c = eye - tf.matmul(tf.transpose(tf.conj(C)), C)
    test_c_norm = tf.norm(test_c)
    W_new = tf.matmul(C, W)
    eye = tf.eye(*tf.Tensor.get_shape(W_new).as_list(), dtype=W_new.dtype)
    test_w_new = eye - tf.matmul(tf.transpose(tf.conj(W_new)), W_new)
    test_w_new_norm = tf.norm(test_w_new)
    init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(visible_device_list=str(0))
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)

with tf.Session(graph=test_graph, config=config) as sess:
    init.run()
    npW, npG, npC, npW_new, np_test_w, np_test_A, np_test_C, np_test_W_new = \
        sess.run([W, G, C, W_new, test_w_norm,
                  test_a_norm, test_c_norm, test_w_new_norm])

    print('unitary W ', np_test_w)
    print('skew    A ', np_test_A)
    print('unitary C ', np_test_C)
    print('unitary Wn', np_test_W_new)
    # plt.imshow(np.abs(np.matmul(np.transpose(np.conj(npW_new)), npW_new)))
    # plt.show()
