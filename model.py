# dcgan by liming @17.7.10

import tensorflow as tf
import os
import numpy as np
import math
import simple_layers as lay

def get_size(size,stride):
	return int(math.ceil(float(size)/float(stride)))


def generate(z, h, w, is_training, reuse, batch_size=64):
	with tf.variable_scope('generator') as scope:
		h1, w1 = get_size(h,2), get_size(w,2)
		h2, w2 = get_size(h1,2), get_size(w1,2)
		h3, w3 = get_size(h2,2), get_size(w2,2)
		h4, w4 = get_size(h3,2), get_size(w3,2)

		fc0 = lay.fully_connect_layer(z, 'g_fc0_lin', h4*w4*512)
		fc0 = tf.reshape(fc0, [-1, h4, w4, 512])
		fc0 = lay.batch_norm_official(fc0, is_training=is_training,reuse=reuse,name='g_bn0')
		fc0 = tf.nn.relu(fc0)

		decon1 = lay.deconv_2d_layer(fc0, 'g_decon1', [5,5,256,512], [batch_size,h3,w3,256], strides=[1,2,2,1])
		decon1 = lay.batch_norm_official(decon1, is_training=is_training,reuse=reuse,name='g_bn1')
		decon1 = tf.nn.relu(decon1)

		decon2 = lay.deconv_2d_layer(decon1, 'g_decon2', [5,5,128,256], [batch_size,h2,w2,128], strides=[1,2,2,1])
		decon2 = lay.batch_norm_official(decon2, is_training=is_training,reuse=reuse,name='g_bn2')
		decon2 = tf.nn.relu(decon2)

		decon3 = lay.deconv_2d_layer(decon2, 'g_decon3', [5,5,64,128], [batch_size,h1,h1,64], strides=[1,2,2,1])
		decon3 = lay.batch_norm_official(decon3, is_training=is_training,reuse=reuse,name='g_bn3')
		decon3 = tf.nn.relu(decon3)

		decon4 = lay.deconv_2d_layer(decon3, 'g_decon4', [5,5,3,64], [batch_size,h,w,3], strides=[1,2,2,1])
		return(tf.nn.tanh(decon4))

def decrim(x, is_training, reuse, batch_size=64):
	with tf.variable_scope('decriminator') as scope:
		conv0 = lay.conv_2d_layer(x, 'd_conv0', [5,5,3,64], strides=[1,2,2,1])
		conv0 = lay.batch_norm_official(conv0, is_training=is_training, reuse=reuse,name = 'd_bn0')
		conv0 = lay.leaky_relu(conv0)

		conv1 = lay.conv_2d_layer(conv0, 'd_conv1', [5,5,64,128], strides=[1,2,2,1])
		conv1 = lay.batch_norm_official(conv1,is_training=is_training,reuse=reuse,name='d_bn1')
		conv1 = lay.leaky_relu(conv1)

		conv2 = lay.conv_2d_layer(conv1, 'd_conv2', [5,5,128,256], strides=[1,2,2,1])
		conv2 = lay.batch_norm_official(conv2,is_training=is_training,reuse=reuse,name='d_bn2')
		conv2 = lay.leaky_relu(conv2)

		conv3 = lay.conv_2d_layer(conv2, 'd_conv3', [5,5,256,512], strides=[1,2,2,1])
		conv3 = lay.batch_norm_official(conv3,is_training=is_training,reuse=reuse,name='d_bn3')
		conv3 = lay.leaky_relu(conv3)

		conv4_flatten = tf.reshape(conv3,[-1,512*6*6])

		fc4 = lay.fully_connect_layer(conv4_flatten,'d_fc4',1)

		return(fc4)





