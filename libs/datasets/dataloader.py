"""
CustomDataset: 						Custom Dataloader for real images using VoxCeleb dataset
CustomDataset_testset_synthetic:	Custom Dataloader for synthetic images for evaluation
CustomDataset_testset_real:			Custom Dataloader for real images for evaluation
"""
import torch
import os
import glob
import cv2
import numpy as np
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import Dataset

from libs.utilities.utils import make_noise

np.random.seed(0)

class CustomDataset(Dataset):

	def __init__(self, dataset_path):
		"""
		VoxCeleb dataset format: id_index/video_index/frames_inverted/latent_codes
		Args:
			dataset_path (string):		 					Path to voxceleb dataset
		"""
		
		self.dataset_path = dataset_path
		self.get_dataset()

		self.transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


	def get_dataset(self):

		ids_path = glob.glob(os.path.join(self.dataset_path, '*/'))
		ids_path.sort()
		if len(ids_path) == 0:
			print('Dataset has no identities in path {}'.format(self.dataset_path))
			exit()
		
		real_images = None; inv_images = None; w = None
		counter_ids = 0; counter_videos = 0
		samples = []
		for i, ids in enumerate(ids_path):
			
			id_index = ids.split('/')[-2]
			videos_path = glob.glob(os.path.join(ids, '*/'))
			videos_path.sort()
			
			for j, video_path in enumerate(videos_path):
				video_index = video_path.split('/')[-2]
				
				if not os.path.exists(os.path.join(video_path, 'inversion')):
					print('Path with inverted latent codes does not exist.')
					exit()
				inv_images_path = glob.glob(os.path.join(video_path, 'inversion', 'frames', '*.png')) # Inverted
				inv_images_path.sort()
				codes_path = glob.glob(os.path.join(video_path, 'inversion', 'latent_codes', '*.npy'))
				codes_path.sort()
				real_images_path = glob.glob(os.path.join(video_path, 'frames_cropped', '*.png'))
				real_images_path.sort()

				dict_sample = {
					'id_index': 		id_index,
					'video_index': 		video_index,
					'real_images':		real_images_path,
					'codes':			codes_path,
					'inv_images':		inv_images_path,
				}
				
				if real_images is None:
					real_images = real_images_path
					w = codes_path
					inv_images = inv_images_path
				else:
					real_images = np.concatenate((real_images, real_images_path), axis=0)
					w = np.concatenate((w, codes_path), axis=0)
					inv_images = np.concatenate((inv_images, inv_images_path), axis=0)					
				counter_videos += 1		
				samples.append(dict_sample)

			counter_ids += 1

		real_images = np.asarray(real_images)
		w = np.asarray(w)
		inv_images = np.asarray(inv_images)
		
		self.real_images = real_images
		self.inv_images = inv_images
		self.w = w
		self.counter_ids = counter_ids
		self.counter_videos = counter_videos

	def get_length(self, train = True):
		return len(self.real_images), self.counter_ids, self.counter_videos
		
	def __len__(self):
		return len(self.real_images)

	def __getitem__(self, index):

		real_image_path = self.real_images[index]
		real_img = Image.open(real_image_path)
		real_img = real_img.convert('RGB')
		real_img = self.transform(real_img)

		inv_image_path = self.inv_images[index]
		inv_img = Image.open(inv_image_path)
		inv_img = inv_img.convert('RGB')
		inv_img = self.transform(inv_img)

		w_file = self.w[index]
		latent_code = np.load(w_file)
		latent_code = torch.from_numpy(latent_code)
		assert latent_code.ndim == 2, 'latent code dimensions should be inject_index x 512 while now is {}'.format(latent_code.shape)

		sample = {
			'real_img': 		real_img,
			'inv_img': 			inv_img,
			'w':				latent_code,
		}
		return sample

class CustomDataset_testset_synthetic(Dataset):

	def __init__(self, synthetic_dataset_path = None, num_samples = None, shuffle = True):
		"""
		VoxCeleb dataset format: id_index/video_index/frames_inverted/latent_codes
		Args:
			synthetic_dataset_path:				path to synthetic latent codes. If None generate random 
			num_samples:						how many samples for validation
			
		"""
		self.shuffle = shuffle
		self.num_samples = num_samples
		self.synthetic_dataset_path = synthetic_dataset_path
		
		
		if self.synthetic_dataset_path is not None:
			z_codes = np.load(self.synthetic_dataset_path)
			z_codes = torch.from_numpy(z_codes)
			self.fixed_source_w =  z_codes[:self.num_samples, :]
			self.fixed_target_w =  z_codes[self.num_samples:2*self.num_samples, :]				
		else:
			self.fixed_source_w = make_noise(self.num_samples, 512, None)
			self.fixed_target_w = make_noise(self.num_samples, 512, None)
			# Save random generated latent codes 
			save_path = './libs/configs/random_latent_codes_{}.npy'.format(2*self.num_samples)
			z_codes = torch.cat((self.fixed_source_w, self.fixed_target_w), dim = 0)
			z_codes = z_codes.detach().cpu().numpy()
			np.save(save_path, z_codes)

		self.transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

	
	def __len__(self):	
		return self.num_samples

	def __getitem__(self, index):
	
		source_w =  self.fixed_source_w[index]
		target_w =  self.fixed_target_w[index]
		sample = {
			'source_w':				source_w,
			'target_w':				target_w
		}
		return sample

class CustomDataset_testset_real(Dataset):

	def __init__(self, dataset_path, suffle = True, num_samples = None):
		"""
		VoxCeleb dataset format: id_index/video_index/frames_inverted/latent_codes
		Args:
			dataset_path (string):		 					Path to voxceleb dataset 
			num_samples:									how many samples for validation
		"""
		self.num_samples = num_samples
		self.dataset_path = dataset_path
		self.suffle = suffle
		
		self.get_dataset()
		self.fixed_target_w = make_noise(self.num_samples, 512, None)

		self.transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

	def get_dataset(self):

		ids_path = glob.glob(os.path.join(self.dataset_path, '*/'))
		ids_path.sort()
		if len(ids_path) == 0:
			print('Dataset has no identities in path {}'.format(self.dataset_path))
			exit()
		
		real_images = None; inv_images = None; w = None
		counter_ids = 0; counter_videos = 0
		samples = []
		for i, ids in enumerate(ids_path):
			
			id_index = ids.split('/')[-2]
			videos_path = glob.glob(os.path.join(ids, '*/'))
			videos_path.sort()
			
			for j, video_path in enumerate(videos_path):
				video_index = video_path.split('/')[-2]	
				if not os.path.exists(os.path.join(video_path, 'inversion')):
					print('Path with inverted latent codes does not exist.')
					exit()		
				codes_path = glob.glob(os.path.join(video_path, 'inversion', 'latent_codes', '*.npy'))
				codes_path.sort()			
				if w is None:
					w = codes_path
				else:
					w = np.concatenate((w, codes_path), axis=0)
				counter_videos += 1		
			counter_ids += 1
		
		self.w = w
		self.counter_ids = counter_ids
		self.counter_videos = counter_videos

		self.w = np.asarray(self.w)
		if self.suffle:
			r = np.random.permutation(len(self.w))
			self.w  = self.w[r.astype(int)]
		
		if self.num_samples < len(self.w):
			self.w = self.w[:self.num_samples]

	def get_length(self):
		return self.num_samples

	def __len__(self):	
		return self.num_samples

	def __getitem__(self, index):
			
		w_file = self.w[index]
		source_w = np.load(w_file)
		source_w = torch.from_numpy(source_w)
		assert source_w.ndim == 2, 'latent code dimensions should be inject_index x 512 while now is {}'.format(source_w.shape)

		target_w =  self.fixed_target_w[index]
		sample = {
			'source_w':				source_w,
			'target_w':				target_w
		}
		return sample
