"""
CustomDataset_paired: 						Custom Dataloader for real paired images training using VoxCeleb dataset
CustomDataset_paired_validation: 			Custom Dataloader for real paired images evaluation using VoxCeleb dataset

"""
import torch
import os
import glob
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class CustomDataset_paired():

	def __init__(self, dataset_path, num_samples = None, max_pairs = 2):
		"""
		VoxCeleb dataset format: id_index/video_index/frames_cropped/*.png
								 id_index/video_index/inversion/frames/*.png
								 id_index/video_index/inversion/latent_codes/*.npy
		Args:
			dataset_path (string):		 					Path to dataset with inverted images. 
			num_samples:									how many samples for validation
			
		"""
		self.dataset_path = dataset_path
		self.num_samples = num_samples
		self.max_pairs = max_pairs
		self.transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

		self.get_dataset()
		
	def get_dataset(self):

		ids_path = glob.glob(os.path.join(self.dataset_path, '*/'))
		ids_path.sort()
		if len(ids_path) == 0:
			print('Dataset has no identities in path {}'.format(self.dataset_path))
			exit()

		sum_images = 0
		counter_ids = 0; counter_videos = 0; counter_imgs = 0
		self.samples_dict = {}
		self.videos_dict = {}
		for i, ids in enumerate(ids_path):
			id_index = ids.split('/')[-2]			
			self.videos_dict.update( {id_index: dict()} )
			videos_path = glob.glob(os.path.join(ids, '*/'))
			videos_path.sort()	
			counter_ids += 1
			for k, video_path in enumerate(videos_path):
				video_id = video_path.split('/')[-2]
				images_path = glob.glob(os.path.join(video_path, 'frames_cropped', '*.png')) # real frames
				images_path.sort()

				if not os.path.exists(os.path.join(video_path, 'inversion')):
					print('Path with inverted latent codes does not exist.')
					exit()

				latent_codes_path = glob.glob(os.path.join(video_path, 'inversion', 'latent_codes', '*.npy'))
				latent_codes_path.sort()
				
				if len(images_path) > 0 and len(latent_codes_path) > 0:
					indices = np.random.permutation(len(images_path))
					images_path = np.asarray(images_path); latent_codes_path = np.asarray(latent_codes_path)
					images_path = images_path[indices.astype(int)]
					latent_codes_path = latent_codes_path[indices.astype(int)]
					dict_ = {
						'num_frames':			len(images_path),
						'frames':				images_path,
						'latent_codes':			latent_codes_path,
					}
					self.videos_dict[id_index].update( {video_id: dict_})

					if len(images_path) >= 2:
						images_path_source = images_path[:self.max_pairs]
						
						for j, image_path in enumerate(images_path_source):
							data = [id_index, video_id, j]
							self.samples_dict.update( {counter_imgs: data} )
							counter_imgs += 1
						counter_videos += 1


		self.num_samples = counter_imgs
		self.counter_ids = counter_ids
		self.counter_videos = counter_videos

	def get_length(self):
		return self.num_samples, self.counter_ids, self.counter_videos
		
	def __len__(self):
		return self.num_samples

	def __getitem__(self, index):

		source_sample = self.samples_dict[index]
		source_id = source_sample[0]
		source_video = source_sample[1]
		source_index = source_sample[2]
		# Get target sample from the same video sequence 
		video_dict = self.videos_dict[source_id][source_video]
		frames_path = video_dict['frames']
		latent_codes_path = video_dict['latent_codes']
		num_frames = video_dict['num_frames']

		target_index = np.random.randint(num_frames, size = 1)[0]
		while target_index == source_index:
			target_index = np.random.randint(num_frames, size = 1)[0]

		source_img_path = frames_path[source_index]
		source_code_path = latent_codes_path[source_index]
		target_img_path = frames_path[target_index]
		target_code_path = latent_codes_path[target_index]


		# Source sample
		source_img = Image.open(source_img_path)
		source_img = source_img.convert('RGB')
		source_img = self.transform(source_img)
		source_latent_code = np.load(source_code_path)
		source_latent_code = torch.from_numpy(source_latent_code)
		# assert source_latent_code.ndim == 2, 'latent code dimensions should be inject_index x 512 while now is {}'.format(source_latent_code.shape)

		# Target sample
		target_img = Image.open(target_img_path)
		target_img = target_img.convert('RGB')
		target_img = self.transform(target_img)
		target_latent_code = np.load(target_code_path)
		target_latent_code = torch.from_numpy(target_latent_code)

		# assert target_latent_code.ndim == 2, 'latent code dimensions should be inject_index x 512 while now is {}'.format(target_latent_code.shape)
		if target_latent_code.ndim == 3:
			target_latent_code = target_latent_code.squeeze(0)
		if source_latent_code.ndim == 3:
			source_latent_code = source_latent_code.squeeze(0)

		sample = {
			'source_img': 					source_img,
			'source_latent_code': 			source_latent_code,
			'target_img': 					target_img,
			'target_latent_code': 			target_latent_code,
			
		}
		return sample


class CustomDataset_paired_validation():

	def __init__(self, dataset_path, num_samples = None):
		"""
		VoxCeleb dataset format: id_index/video_index/frames_cropped/*.png
								 id_index/video_index/inversion/frames/*.png
								 id_index/video_index/inversion/latent_codes/*.npy
		Args:
			dataset_path (string):		 					Path to dataset with inverted images. 
			num_samples:									how many samples for validation
			
		"""

		self.dataset_path = dataset_path
		self.num_samples = num_samples
		self.transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

		self.get_dataset()

	def get_dataset(self):

		ids_path = glob.glob(os.path.join(self.dataset_path, '*/'))
		ids_path.sort()
		if len(ids_path) == 0:
			print('Dataset has no identities in path {}'.format(self.dataset_path))
			exit()

		sum_images = 0
		counter_ids = 0; counter_videos = 0; counter_imgs = 0
		self.samples_dict = {}
		self.videos_dict = {}
		for i, ids in enumerate(ids_path):
			id_index = ids.split('/')[-2]			
			self.videos_dict.update( {id_index: dict()} )
			videos_path = glob.glob(os.path.join(ids, '*/'))
			videos_path.sort()	
			counter_ids += 1
			for k, video_path in enumerate(videos_path):
				video_id = video_path.split('/')[-2]
				images_path = glob.glob(os.path.join(video_path, 'frames_cropped', '*.png')) # real frames
				images_path.sort()
				if not os.path.exists(os.path.join(video_path, 'inversion')):
					print('Path with inverted latent codes does not exist.')
					exit()
				latent_codes_path = glob.glob(os.path.join(video_path, 'inversion', 'latent_codes', '*.npy'))
				latent_codes_path.sort()

				dict_ = {
					'num_frames':			len(images_path),
					'frames':				images_path,
					'latent_codes':			latent_codes_path
				}
				self.videos_dict[id_index].update( {video_id: dict_})

				if len(images_path) >= 2:
					for j, image_path in enumerate(images_path):
						target_index = np.random.randint(len(images_path), size = 1)[0]
						while target_index == j:
							target_index = np.random.randint(len(images_path), size = 1)[0]
						data = [id_index, video_id, j, target_index]
						self.samples_dict.update( {counter_imgs: data} )
						counter_imgs += 1
					counter_videos += 1

		self.num_samples = counter_imgs
		self.counter_ids = counter_ids
		self.counter_videos = counter_videos

	def get_length(self):	
		return self.num_samples
	
	def __len__(self):
		return self.num_samples

	def __getitem__(self, index):

		source_sample = self.samples_dict[index]
		source_id = source_sample[0]
		source_video = source_sample[1]
		source_index = source_sample[2]
		target_index = source_sample[3]
		# Get target sample from the same video sequence 
		video_dict = self.videos_dict[source_id][source_video]
		frames_path = video_dict['frames']
		latent_codes_path = video_dict['latent_codes']
		num_frames = video_dict['num_frames']

		source_img_path = frames_path[source_index]
		source_code_path = latent_codes_path[source_index]
		target_img_path = frames_path[target_index]
		target_code_path = latent_codes_path[target_index]


		# Source sample
		source_img = Image.open(source_img_path)
		source_img = source_img.convert('RGB')
		source_img = self.transform(source_img)
		source_latent_code = np.load(source_code_path)
		source_latent_code = torch.from_numpy(source_latent_code)
		# assert source_latent_code.ndim == 2, 'latent code dimensions should be inject_index x 512 while now is {}'.format(source_latent_code.shape)

		# Target sample
		target_img = Image.open(target_img_path)
		target_img = target_img.convert('RGB')
		target_img = self.transform(target_img)
		target_latent_code = np.load(target_code_path)
		target_latent_code = torch.from_numpy(target_latent_code)

		# assert target_latent_code.ndim == 2, 'latent code dimensions should be inject_index x 512 while now is {}'.format(target_latent_code.shape)
		if target_latent_code.ndim == 3:
			target_latent_code = target_latent_code.squeeze(0)
		if source_latent_code.ndim == 3:
			source_latent_code = source_latent_code.squeeze(0)

		sample = {
			'source_img': 					source_img,
			'source_latent_code': 			source_latent_code,
			'target_img': 					target_img,
			'target_latent_code': 			target_latent_code,
			
		}
		return sample
