#!/usr/bin/env python3
# random erasing in tomogram, generate N blocks in image, in order to train a robuster model
# use mean value instead of random value


import fire
import mrcfile
import numpy as np



def random_erasing(img, M, N1):
	width = img.shape[0]
	height = img.shape[1]

	square = 15
	for attempt in range(N1):
		x1 = np.random.randint(0, width - square)
		y1 = np.random.randint(0, height - square)
		img[x1: x1+square, y1: y1+square] = M
	return img


def random_erasing_3d(img, M, N2):
	width, height, layer = img.shape[0], img.shape[1], img.shape[2]

	cube = 10
	for attempt in range(N2):
		x1 = np.random.randint(0, width-cube)
		y1 = np.random.randint(0, height-cube)
		z1 = np.random.randint(0, layer-cube)
		img[x1:x1+cube, y1:y1+cube, z1:z1+cube] = M
	return img


def apply(mrc, N1=40, N2=120, output=None):
	'''
	param N1: number of 2d blocks in image with size 15*15
	param N2: number of 3d blocks in image with size 10*10*10
	'''

	with mrcfile.open(mrc) as mrc_file:
		data = mrc_file.data
		data.flags.writeable = True
		data = data.transpose(2, 1, 0)
	mean = np.mean(data)
	mrc_random_erased = np.zeros_like(data)

	for i in range(data.shape[2]):
		mrc_random_erased[:, :, i] = random_erasing(data[:, :, i], mean, N1)
	mrc_random_erased = random_erasing_3d(mrc_random_erased, mean, N2)

	mrc_random_erased = mrc_random_erased.transpose(2, 1, 0)

	if output is None:
		output = mrc.replace('.mrc', '_era'+str(N1)+'-'+str(N2)+'.mrc')
	with mrcfile.new(output, overwrite=True) as mrc_file:
		mrc_file.set_data(mrc_random_erased.astype(np.float32))
	
	print('random erasing done, output:', output)


if __name__ == '__main__':
	fire.Fire(apply)
