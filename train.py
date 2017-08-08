#dcgan by liming @17.7.10
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob
from model import *
from pre_data import *


Image_h = 96
Image_w = 96
Sample_num = 30000
Epoch_num = 400
Batch_size = 64
G_learnrate = 1e-3
D_learnrate = 1e-3
tensorboad_dir = 'logs6'  
# Data_dir = 'faces'
Data_dir = 'CelebA/images'
Data_pattern = '*.jpg'


def optimizer(loss, learning_rate, vlist=None, name=None):
    with tf.variable_scope(name):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name=name + '/Adam')
        return opt.minimize(loss, var_list=vlist, name=name + '/opt')

def draw_img(x):
	pass

def __main__():

	#noise input
	noise_input = tf.placeholder(tf.float32, shape = [None,100], name='noise')
	noise_sample_input = tf.placeholder(tf.float32, shape = [None,100], name='noise')
	#real data input
	image_input = tf.placeholder(tf.float32, shape = [None,Image_h,Image_w,3], name='image')
	
	#generate G
	G = generate(noise_input, Image_h, Image_w, True, None, batch_size=Batch_size)
	# param of G
	G_vars = tf.trainable_variables()

	G_sample = generate(noise_sample_input, Image_h, Image_w, False, True, batch_size=Batch_size)
	img_sample = restruct_image(G_sample,Batch_size)
	tf.summary.image('generated image',img_sample,Batch_size)
	#decrim 
	D = decrim(image_input, True, None, batch_size=Batch_size)
	# param of d
	D_vars = []
	for item in tf.trainable_variables():
		if item not in G_vars:
			D_vars.append(item)


	d_real = D
	d_fake = decrim(G, True, True)

	loss_train_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))) \
					+ tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
	tf.summary.scalar('d_loss',loss_train_D)

	loss_train_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
	tf.summary.scalar('g_loss',loss_train_G)
	# loss_train_G = d_fake
	# loss_train_D = -(d_real + d_fake)
    # loss_train_G = (1 / 2) * (d_fake - 1) ** 2
    # loss_train_D = (1 / 2) * (d_real - 1) ** 2 + (1 / 2) * (d_fake) ** 2
    # 训练生成器的优化器，对应判别器为不可训练
	g_optimizer = optimizer(loss_train_G, G_learnrate, G_vars, name='opt_train_G')
    # 训练判别器的优化器
	d_optimizer = optimizer(loss_train_D, D_learnrate, D_vars, name='opt_train_D')

	# noise_sample = np.random.uniform(-1,1,[Batch_size,100]).astype('float32')
    #==============================Start training=============================
	with tf.Session() as sess:
		#=====tensorboard=============
		merged_summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(tensorboad_dir, sess.graph)
		#=============================
		sess.run(tf.global_variables_initializer())
		image_list = get_datalist(Data_dir,Data_pattern)[:Sample_num]
		image_len = int(len(image_list))
		batch_num = int(image_len/Batch_size)
		count = 0
		for e in range(Epoch_num):

			for idx in range(batch_num):
    			#prepare data
				z = np.random.normal(0,1,[Batch_size,100]).astype('float32')
				img_batch_list = image_list[idx*Batch_size:(idx+1)*Batch_size]
				img_batch = get_image(img_batch_list,Batch_size,Image_h,Image_w)

				_, d_loss = sess.run([d_optimizer,loss_train_D],
    										feed_dict={
    											noise_input: z,
    											image_input: img_batch,

    										})

				_, g_loss = sess.run([g_optimizer,loss_train_G],
    										feed_dict={
    											noise_input: z,
    											image_input: img_batch,

    										})

				print("epoch: %d batch: %d  gloss:%.4f dloss:%.4f" %
                                         (e + 1, idx, g_loss, d_loss))

				

				


				if idx%10==0:
					count = count + 1
					noise_sample = np.random.normal(0,1,[Batch_size,100]).astype('float32')

					sumarry_all = sess.run(merged_summary_op,feed_dict={
												noise_sample_input: noise_sample,
    											noise_input: z,
    											image_input: img_batch,

    										})
					summary_writer.add_summary(sumarry_all,count)
					

				


# =================================================================
if __name__ == "__main__":
    __main__()
