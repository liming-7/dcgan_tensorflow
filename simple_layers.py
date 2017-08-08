# -*-encoding: utf8-*-

import tensorflow as tf

from tensorflow.contrib.layers import batch_norm
import os


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def shared_variable(name, initializer, dtype=tf.float32):
    # 获取一个共享变量（不存在则新建，存在则返回）
    try:
        with tf.variable_scope(name):
            v = tf.get_variable('var', dtype=dtype, initializer=initializer)
            variable_summaries(v, name + '/var')
    except ValueError as e:
        if str(e).find('reuse') == -1:
            raise
        with tf.variable_scope(name, reuse=True):
            v = tf.get_variable('var')
    return v


def fully_connect_layer(x, name, output_n, stddev=0.2, dtype=tf.float32, get_w_and_b=False):
    # 创建一个全连接层
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        input_n = shape[1]
        w = shared_variable('weight', tf.truncated_normal([input_n, output_n], stddev=stddev), dtype=dtype)
        b = shared_variable('bias', tf.zeros([output_n]), dtype=dtype)
        z = tf.matmul(x, w, name=name + '/fc') + b
    if get_w_and_b:
        return z, w, b
    else:
        return z


def conv_2d_layer(x,
                  name,
                  filter_shape,
                  stddev=0.2,
                  strides=(1, 1, 1, 1),
                  padding='SAME',
                  dtype=tf.float32,
                  get_w_and_b=False):
    # 创建一个2d卷积层
    with tf.variable_scope(name):
        output_n = filter_shape[3]
        filter_w = shared_variable('filter', tf.truncated_normal(filter_shape, stddev=stddev), dtype=dtype)
        b = shared_variable('bias', tf.zeros([output_n]), dtype=dtype)
        z = tf.nn.conv2d(x, filter_w, strides, padding, name=name + '/conv') + b
    if get_w_and_b:
        return z, filter_w, b
    else:
        return z


def deconv_2d_layer(x,
                    name,
                    filter_shape,
                    output_shape,
                    stddev=0.2,
                    strides=(1, 1, 1, 1),
                    padding='SAME',
                    dtype=tf.float32,
                    get_w_and_b=False):
    with tf.variable_scope(name):
        output_n = filter_shape[2]
        filter_w = shared_variable('filter', tf.truncated_normal(filter_shape, stddev=stddev), dtype=dtype)
        b = shared_variable('bias', tf.zeros([output_n]), dtype=dtype)
        z = tf.nn.conv2d_transpose(x, filter_w, output_shape, strides, padding, name=name + '/deconv') + b
    if get_w_and_b:
        return z, filter_w, b
    else:
        return z


def expand(x):
    return tf.reshape(x, [1, 1, 1, -1])


def batch_norm_unofficial(x, g=None, b=None, u=None, s=None, a=1., e=1e-8, name='batchnorm'):
    with tf.variable_scope(name):
        if len(x.get_shape()) == 4:
            if u is not None and s is not None:
                b_u = expand(u)
                b_s = expand(s)
            else:
                b_u = expand(tf.reduce_mean(x, axis=[0, 1, 2]))
                b_s = expand(tf.reduce_mean(tf.square(x - b_u), axis=[0, 1, 2]))
            if a != 1:
                b_u = (1. - a) * 0. + a * b_u
                b_s = (1. - a) * 1. + a * b_s
            x = (x - b_u) / tf.sqrt(b_s + e)
            if g is not None and b is not None:
                x = x * expand(g) + expand(b)
        elif len(x.get_shape()) == 2:
            if u is None and s is None:
                u = tf.reduce_mean(x, axis=0)
                s = tf.reduce_mean(tf.square(x - u), axis=0)
            if a != 1:
                u = (1. - a) * 0. + a * u
                s = (1. - a) * 1. + a * s
            x = (x - u) / tf.sqrt(s + e)
            if g is not None and b is not None:
                x = x * g + b
        else:
            raise NotImplementedError
    return x


def batch_norm_official(x, is_training, reuse, trainable=True, decay=0.99, name='batchnorm'):
    if not reuse and 'batchnorm' == name:
        print('Waring: if this batchnorm layer not mean to reuse, give it an unique name.')
    with tf.variable_scope(name):
        x = batch_norm(x, decay=decay, updates_collections=None, is_training=is_training,
                       reuse=reuse, trainable=trainable, scope=name)
    return x


def leaky_relu(x, alpha=0.1, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(tf.multiply(alpha, x, name=name + 'lrelu/add'), x, name=name + 'lrelu/maxmium')


def conv_concat(x, y, name='conv_concat'):
    with tf.variable_scope(name):
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return tf.concat(3, [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], name='conv_concat')


def visualization():
    if os.path.exists('./graphics'):
        __import__('shutil').rmtree('./graphics')
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./graphics', session.graph)
        summary = session.run(merged)
        writer.add_summary(summary)
    os.system('tensorboard --logdir=./graphics')


def __main__():
    x = shared_variable('x', tf.zeros([3, 5]))
    fc1 = fully_connected_layer(x, 'fc1', 10)

    y = shared_variable('y', tf.zeros([3, 10, 10, 6]))
    conv1, conv1_b, conv1_w = conv_2d_layer(y, 'conv1', [5, 5, 6, 12], padding='VALID', get_w_and_b=True)

    bn1 = batch_norm_official(conv1, True, False, name='bn1')
    bn2 = batch_norm_official(conv1, False, False, name='bn2')
    bn1_2 = batch_norm_official(conv1, False, True, name='bn1')

    deconv1 = deconv_2d_layer(bn2, 'deconv1', [5, 5, 6, 12], [3, 10, 10, 6], padding='VALID')

    visualization()


if __name__ == '__main__':
    __main__()
