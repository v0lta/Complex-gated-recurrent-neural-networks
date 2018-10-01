import tensorflow as tf
from IPython.core.debugger import Tracer
debug_here = Tracer()


def complex_conv1D(h, filter_width, depth, stride, padding, scope='', reuse=None):
    """
    Implement W*h by using the distributive property of the convolution.
    """
    in_channels = h.get_shape().as_list()[2]

    with tf.variable_scope('complex_conv1D' + scope, reuse=reuse):
        if 0:
            print("REAL!")
            Wstack = tf.get_variable('cmplx_conv_weights',
                                     [filter_width, in_channels, depth],
                                     initializer=tf.glorot_normal_initializer())
            return tf.nn.conv1d(tf.abs(h), Wstack, stride=stride, padding=padding)
        else:
            Wstack = tf.get_variable('cmplx_conv_weights',
                                     [filter_width, in_channels, depth, 2],
                                     initializer=tf.glorot_normal_initializer())
            f_real = Wstack[:, :, :, 0]
            f_imag = Wstack[:, :, :, 1]
            x = tf.real(h)
            y = tf.imag(h)

            # xfr = tf.nn.conv1d(value=x, filters=f_real, stride=stride, padding=padding)
            # yfr = tf.nn.conv1d(value=y, filters=f_real, stride=stride, padding=padding)
            # xfi = tf.nn.conv1d(value=x, filters=f_imag, stride=stride, padding=padding)
            # yfi = tf.nn.conv1d(value=y, filters=f_imag, stride=stride, padding=padding)
            # return tf.complex(xfr - yfi, xfi + yfr)

            cat_x = tf.concat([x, y], axis=-1)
            cat_kernel_4_real = tf.concat([f_real, -f_imag], axis=-2)
            cat_kernel_4_imag = tf.concat([f_imag, f_real], axis=-2)
            cat_kernels_4_complex = tf.concat([cat_kernel_4_real,
                                               cat_kernel_4_imag],
                                              axis=-1)
            conv = tf.nn.conv1d(value=cat_x, filters=cat_kernels_4_complex,
                                stride=stride, padding=padding)
            conv_2 = tf.split(conv, axis=-1, num_or_size_splits=2)
            return tf.complex(conv_2[0], conv_2[1])


def complex_max_pool1d(h, ksize, strides, padding, scope=None):
    """
    Complex pooling.
    """
    with tf.variable_scope('complex_max_pool1d' + scope):
        real_pool = tf.nn.max_pool(tf.expand_dims(tf.real(h), 1), ksize, strides, padding)
        imag_pool = tf.nn.max_pool(tf.expand_dims(tf.imag(h), 1), ksize, strides, padding)
        return tf.complex(real_pool, imag_pool)
