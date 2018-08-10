import torch
import numpy as np
from timeit import default_timer
from dataset.general.serialize import data2pa, pa2np
from BASIC_LIST.basic import groupby
import os
import cv2



class train_loader:
	def __init__(self, folder, minibatch=32):
		r'''
        Loading data.
        '''
		end = default_timer()		
		print("Loader Process Begin.")
		X = []
		file_list = os.listdir(folder)

		for each in file_list:
			if ('.npy' in each):
				npy = np.load(os.path.join(folder, each))
				print(npy.shape)
				X.append(npy)

		self.X = np.concatenate(X).astype(np.float32)[:96000]

		print('loading time: {}, final X shape: {}'.format(
			default_timer()-end, self.X.shape))
		self.minibatch = minibatch

	def get(self):
		indices = np.random.permutation(len(self.X)).tolist()
		groups = groupby(indices, self.minibatch, key='mini')
		for group in groups:
			yield self.X[group]



def resize_128(img, fig_size=(128, 128)):
	'''
	@img should be a numpy ndarray with shape (H, W, C)
	@fig_size should be a tuple (W, H), but as we need square image, 
	 it doen't matter.
	'''
	return cv2.resize(img, fig_size, interpolation=cv2.INTER_AREA)
