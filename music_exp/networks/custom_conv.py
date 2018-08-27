import tensorflow as tf


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
                                     initializer=tf.glorot_uniform_initializer())
            return tf.nn.conv1d(tf.abs(h), Wstack, stride=stride, padding=padding)
        else:
            Wstack = tf.get_variable('cmplx_conv_weights',
                                     [filter_width, in_channels, depth, 2],
                                     initializer=tf.glorot_uniform_initializer())
            A = Wstack[:, :, :, 0]
            B = Wstack[:, :, :, 1]
            x = tf.real(h)
            y = tf.imag(h)

            xA = tf.nn.conv1d(value=x, filters=A, stride=stride, padding=padding)
            yA = tf.nn.conv1d(value=y, filters=A, stride=stride, padding=padding)
            xB = tf.nn.conv1d(value=x, filters=B, stride=stride, padding=padding)
            yB = tf.nn.conv1d(value=y, filters=B, stride=stride, padding=padding)
            return tf.complex(xA - yB, xB + yA)
