'''
api of layers
'''
import os
import numpy as np
from glob import glob
from PIL import Image

def get_datalist(data_dir,data_pattern):
	image_list = glob(os.path.join(data_dir,data_pattern))
	return (image_list)

def get_image(image_list,batch_size,img_h,img_w):
	image_batch = []
	for img in image_list:
		data = Image.open(img)
		data = data.resize((img_h,img_w))
		data = np.array(data)
		data = data.astype('float32')/127.5 -1
		image_batch.append(data)
	return (image_batch)


def restruct_image(x,batch_size):
	image_batch = []
	for k in range(batch_size):
		data = x[k,:,:,:]
		data = (data+1)*127.5
		# data = np.clip(data,0,255).astype(np.uint8)
		image_batch.append(data)
	return (image_batch)