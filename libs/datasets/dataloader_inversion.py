
import torch
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils as torch_utils

class DatasetInversion(torch.utils.data.Dataset):

	def __init__(self, root_path, num_images = None):
		"""
		Args:
			root_path: VoxCeleb dataset images are saved as: id_index/video_id/frames_path/*.png
		"""
		self.root_path = root_path
		self.images_files = None
		
		ids_path = glob.glob(os.path.join(root_path, '*/'))
		ids_path.sort()
		count_videos = 0
		# print('Dataset has {} identities'.format(len(ids_path)))
		for i, id_path in enumerate(ids_path):
			id_index = id_path.split('/')[-2]
			videos_path = glob.glob(os.path.join(id_path, '*/'))
			videos_path.sort()
			count_videos += len(videos_path)
			for j, video_path in enumerate(videos_path):
				images_files_tmp = glob.glob(os.path.join(video_path, 'frames_cropped', '*.png'))
				images_files_tmp.sort()
				if self.images_files is None:
					self.images_files = images_files_tmp
				else:
					self.images_files = np.concatenate((self.images_files, images_files_tmp))
		
		if num_images is not None:
			self.images_files = self.images_files[:num_images]
		self.images_files.sort()

		self.len_images = self.get_length()
		self.indices = [ x for x in range(self.len_images) ]
		self.indices_temporal = self.indices.copy()

		print('Dataset has {} identities, {} videos and {} frames'.format(len(ids_path), count_videos, len(self.images_files)))

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.images_files)

	def __getitem__(self, index):

		img_name = self.images_files[index]
		tmp = img_name.split('/')
		filenames = tmp[-1]
		video_indices = tmp[-3]
		id_indices = tmp[-4]

		images = self.image_to_tensor(img_name)	

		out_dict = {
			'images':			images,
			'filenames':		filenames,
			'id_indices':		id_indices,
			'video_indices':	video_indices
		}
		
		return out_dict
		
	def get_length(self):

		return len(self.images_files)
		
	def get_sample(self, idx):
		
		img_name = self.images_files[idx]
		tmp = img_name.split('/')
		filename = tmp[-1]
		video_index = tmp[-3]
		id_index = tmp[-4]

		image = self.image_to_tensor(img_name)	

		return image, filename, id_index, video_index
		
	def get_batch(self, batch_size = 2):
	
		images = torch.zeros(batch_size, 3, 256, 256)
		filenames = []
		id_indices = []; video_indices = []
		for i in range(int(batch_size)):
			if len(self.indices_temporal)>0:
				index_pick = self.indices_temporal[-1]
				image, filename, id_index, video_index = self.get_sample(index_pick)
				self.indices_temporal.pop(-1)
				filenames.append(filename)
				id_indices.append(id_index)
				video_indices.append(video_index)
				images[i] = image
		
		out_dict = {
			'images':			images,
			'filenames':		filenames,
			'id_indices':		id_indices,
			'video_indices':	video_indices
		}
		return out_dict
	
	'Transform images to tensor [-1,1]. Generators space'
	def image_to_tensor(self, image_file):
		max_val = 1
		min_val = -1
		# for image_file in image_files:
		image = cv2.imread(image_file, cv2.IMREAD_COLOR) # BGR order!!!!
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('uint8')


		if image.shape[0]>256:
			image, _ = image_resize(image, 256)
		
		image_tensor = torch.tensor(np.transpose(image,(2,0,1))).float().div(255.0)	
		image_tensor = image_tensor * (max_val - min_val) + min_val

		return image_tensor