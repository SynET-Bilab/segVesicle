#! usr/bin/env python3
# random erasing in .mrc data, generate N blocks in image, in order to train a robuster model
# use mean value instead of random value

# 3d random erasing plays different role with 2d one, 2d simulate low-quality image while 3d simulate omega vesciles and others similar
# 

import mrcfile
import numpy as np
import argparse

N1 = 40 # N blocks in image(for 2d)
N2 = 120 # N blocks (for 3d)


def random_erasing(img, M):
	width = img.shape[0]
	height = img.shape[1]

	for attempt in range(N1):
		se = 15*15 #15*15 in bin8 files
		a = int(np.round(np.sqrt(se)))
		if a < min(width, height):
			x1 = np.random.randint(0, width - a)
			y1 = np.random.randint(0, height - a)
			img[x1: x1+a, y1: y1+a] = 0
	return img


def random_erasing_3d(img, M):
	width, height, layer = img.shape[0], img.shape[1], img.shape[2]

	for attempt in range(N2):
		se = 10*10
		a = int(np.round(np.sqrt(se)))
		if a < min(width, height, layer):
			x1 = np.random.randint(0, width-a)
			y1 = np.random.randint(0, height-a)
			z1 = np.random.randint(0, layer-a)
			img[x1:x1+a, y1:y1+a, z1:z1+a] = 0
	return img


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--tomo_file', type=str, default=None, help='Your original tomo')
	parser.add_argument('--result_tomo', type=str, default=None, help='result tomo')
	args = parser.parse_args()

	with mrcfile.open(args.tomo_file) as s:
		image = s.data # axis: z, y, x
		image.flags.writeable = True
		image = image.transpose(2, 1, 0) # (x, y, z)
	mean = image.mean() # use mean value instead of random value
	image_ran_era = np.zeros(image.shape)

	for i in range(image.shape[2]):
		image_ran_era[:, :, i] = random_erasing(image[:, :, i], M = mean)
	
	image_ran_era = random_erasing_3d(image_ran_era, M = mean)

	image_ran_era = image_ran_era.transpose(2, 1, 0)

	with mrcfile.new(args.result_tomo, overwrite=True) as m:
		m.set_data(image_ran_era.astype(np.float32))
    
	print("already erasing")
    
    
    
