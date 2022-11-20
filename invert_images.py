"""
Invert VoxCeleb dataset using Encoder4Editing method https://github.com/omertov/encoder4editing

Inputs:
	--input_path:  	Path to voxceleb dataset. The dataset format should be: id_index/video_id/frames_cropped/*.png

Inverted images will be saved in input_path as: 
					id_index/video_id/inversion/frames/*.png
					id_index/video_id/inversion/latent_codes/*.npy

python invert_images.py --input_path /datasets/VoxCeleb1/VoxCeleb1_test
"""

import os
import numpy as np
import torch
from torchvision import utils as torch_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

from libs.gan.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.datasets.dataloader_inversion import DatasetInversion
from libs.gan.encoder4editing.psp_encoders import Encoder4Editing


parser = ArgumentParser()

parser.add_argument("--input_path",  required = True, help='Path to VoxCeleb dataset')

parser.add_argument("--generator_path", 	default = './pretrained_models/stylegan-voxceleb.pt', 	help='Path to generator model')
parser.add_argument("--encoder_path",		default = './pretrained_models/e4e-voxceleb.pt', 		help='Path to encoder model')
parser.add_argument("--batch_size", 		default = 4, 	help='batch size')
parser.add_argument("--image_resolution", 	default = 256, 	help='Path to generator model')
parser.add_argument("--channel_multiplier", default = 1, 	help='Path to generator model')


def make_path(path):
	if not os.path.exists(path):
		os.makedirs(path)

class Inversion(object):

	def __init__(self):

		args = parser.parse_args()
		self.input_path = args.input_path

		self.output_path = self.input_path # output_path same with the input_path"
		self.generator_path = args.generator_path
		self.encoder_path = args.encoder_path
		self.channel_multiplier = args.channel_multiplier
		self.batch_size = args.batch_size
		self.image_resolution = args.image_resolution
		

	def load_networks(self):
		print('----- Load encoder from {} -----'.format(self.encoder_path))
		
		self.encoder = Encoder4Editing(50, 'ir_se', self.image_resolution)
		ckpt = torch.load(self.encoder_path)
		self.encoder.load_state_dict(ckpt['e']) 
		self.encoder.cuda().eval()
		
		print('----- Load generator from {} -----'.format(self.generator_path))
		self.generator  = StyleGAN2Generator(self.image_resolution, 512, 8, channel_multiplier = self.channel_multiplier)
		self.generator.load_state_dict(torch.load(self.generator_path)['g_ema'], strict = False)
		self.generator.cuda().eval()
		
		self.truncation = 0.7
		self.trunc = self.generator.mean_latent(4096).detach().clone()

	def configure_dataset(self):
		self.dataset = DatasetInversion(self.input_path, num_images=None)
		
		self.dataloader = DataLoader(self.dataset,
									batch_size=self.batch_size,
									shuffle=False,
									num_workers=1,
									drop_last=False)

	def run_inversion_dataset(self):
		
		if not os.path.exists(self.output_path):
			os.makedirs(self.output_path, exist_ok=True)
		
		self.configure_dataset()
		self.load_networks()
		step = 0

		for batch_idx, batch in enumerate(tqdm(self.dataloader)): 

			sample_dict = batch 
			filenames = sample_dict['filenames']
			id_indices = sample_dict['id_indices']
			video_indices = sample_dict['video_indices']
			images = sample_dict['images'].cuda()
			
			with torch.no_grad():
				latent_codes = self.encoder(images)
				inverted_images, _ = self.generator([latent_codes], input_is_latent=True, return_latents = False, truncation= self.truncation, truncation_latent=self.trunc)
			
			
			# Save inverted images and latent codes
			for i in range(len(id_indices)):
				output_path_local = os.path.join(self.output_path, id_indices[i], video_indices[i], 'inversion')
				make_path(output_path_local)
				make_path(os.path.join(output_path_local, 'frames'))
				save_dir = os.path.join(output_path_local, 'frames', filenames[i])
				grid = torch_utils.save_image(
					inverted_images[i],
					save_dir,
					normalize=True,
					range=(-1, 1),
				)

				# Latent code
				latent_code = latent_codes[i].detach().cpu().numpy()
				make_path(os.path.join(output_path_local, 'latent_codes'))				
				latent_filename = filenames[i].split('.')[0]
				latent_filename = latent_filename + '.npy'
				save_dir = os.path.join(output_path_local, 'latent_codes', latent_filename)	
				np.save(save_dir, latent_code)

			step += self.batch_size
			
if __name__ == "__main__":

	inversion = Inversion()
	inversion.run_inversion_dataset()

	
				

