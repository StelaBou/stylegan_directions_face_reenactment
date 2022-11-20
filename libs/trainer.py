import os
import json
import torch
import time
import numpy as np
import pdb
import cv2
import wandb
from torch import nn
from torch.utils.data import DataLoader

from libs.utilities.utils import save_arguments_json, make_path, make_noise
from libs.DECA.estimate_DECA import DECA_model
from libs.models.direction_matrix import DirectionMatrix
from libs.gan.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.utilities.utils_train import Utilities_train
from libs.utilities.generic import calculate_shapemodel, generate_image
from libs.datasets.dataloader_paired import CustomDataset_paired
from libs.configs.config_models import *

class Trainer(object):

	def __init__(self, args):
		self.use_cuda = True
		self.device = 'cuda'
		self.args = args
		self.initialize_arguments(args)
		################# Initialize output paths #################
		make_path(self.output_path)
		self.log_dir = os.path.join(self.output_path, 'logs')
		make_path(self.log_dir)	
		self.models_dir = os.path.join(self.output_path, 'models')
		make_path(self.models_dir)
		self.images_dir = os.path.join(self.log_dir, 'images')
		make_path(self.images_dir)
		self.images_reenact_dir = os.path.join(self.log_dir, 'reenactment')
		make_path(self.images_reenact_dir)
		####################################################################	
		### save arguments file with params
		save_arguments_json(args, self.output_path, 'arguments.json')
			
	def initialize_arguments(self, args):
		
		self.output_path = args['experiment_path']
		self.use_wandb = args['use_wandb']
		self.log_images_wandb = args['log_images_wandb']
		self.project_wandb = args['project_wandb']
		self.resume_training_model = args['resume_training_model']
			
		self.image_resolution = args['image_resolution']
		self.dataset_type = args['dataset_type'] 
		if self.dataset_type == 'voxceleb' and self.image_resolution == 256:
			self.gan_weights = stylegan2_voxceleb_256['gan_weights'] 
			self.channel_multiplier = stylegan2_voxceleb_256['channel_multiplier']
		elif self.dataset_type == 'ffhq' and self.image_resolution == 256:
			self.gan_weights = stylegan2_ffhq_256['gan_weights'] 
			self.channel_multiplier = stylegan2_ffhq_256['channel_multiplier']
		elif self.dataset_type == 'ffhq' and self.image_resolution == 1024:
			self.gan_weights = stylegan2_ffhq_1024['gan_weights'] 
			self.channel_multiplier = stylegan2_ffhq_1024['channel_multiplier']
		else:
			print('Incorect dataset type {} and image resolution {}'.format(self.dataset_type, self.image_resolution))

		self.synthetic_dataset_path = args['synthetic_dataset_path']
		self.train_dataset_path = args['train_dataset_path']
		self.test_dataset_path = args['test_dataset_path'] 
		
		self.disentanglement_50 = args['disentanglement_50']
		self.w_plus = args['w_plus']
		self.num_layers_shift = args['num_layers_shift']
		self.training_method = args['training_method']
		self.learned_directions = args['learned_directions']
		self.shift_scale = args['shift_scale'] 
		self.min_shift = args['min_shift'] 
		self.lr = args['lr'] 
		self.batch_size = args['batch_size'] 
		self.test_batch_size = args['test_batch_size']
		self.workers = args['workers']
		self.lambda_identity = args['lambda_identity'] 
		self.lambda_perceptual = args['lambda_perceptual']
		self.lambda_shape = args['lambda_shape']
		self.lambda_mouth_shape = args['lambda_mouth_shape']
		self.lambda_eye_shape = args['lambda_eye_shape']
		self.lambda_w_reg = args['lambda_w_reg']
		self.n_steps = args['max_iter']
		self.validation_samples = args['validation_samples']

	def load_models(self):
		###################### Initialize models ###########################
		print('***************************** Load DECA model  *****************************')
		self.deca = DECA_model(self.device)	
		####################################################################


		####################### Initialize Direction Matrix A #######################
		print('***************************** Initialize Direction Matrix A model *****************************')
		self.dim_z = 512
		self.A_matrix = DirectionMatrix(shift_dim = self.dim_z,
				input_dim = self.learned_directions,
				w_plus = self.w_plus, num_layers = self.num_layers_shift)
		self.A_matrix = self.A_matrix.cuda()
		####################################################################

		####################### Initialize generator #######################
		print('*****************************Load generator from {} *****************************'.format(self.gan_weights))
		self.G = StyleGAN2Generator(self.image_resolution, 512, 8, channel_multiplier= self.channel_multiplier)
		if self.image_resolution == 256:
			self.G.load_state_dict(torch.load(self.gan_weights)['g_ema'], strict = False)
		else:
			self.G.load_state_dict(torch.load(self.gan_weights)['g_ema'], strict = True)
		self.G.cuda().eval()
		self.truncation = 0.7
		self.trunc = self.G.mean_latent(4096).detach().clone()
		####################################################################
		
	def initialize_utilities_train(self):
		params_train = self.args
		if self.use_wandb:
			self.utils_train = Utilities_train(params_train, self.deca, self.images_dir, self.images_reenact_dir, self.truncation, self.trunc, wandb)
		else:
			self.utils_train = Utilities_train(params_train, self.deca, self.images_dir, self.images_reenact_dir, self.truncation, self.trunc, None)

	def initialize_wandb(self):
		#########################
		config = self.args
		wandb.init(
			project= self.project_wandb,
			config=config,
		)
		name = self.output_path.split('/')[-1]
		wandb.run.name = name
		wandb.watch(self.A_matrix, log="all", log_freq=100)
		#######################
	
	def train(self):

		self.load_models()
		if self.use_wandb:
			self.initialize_wandb()
		self.initialize_utilities_train()
		self.utils_train.configure_dataset()

		self.A_matrix.train() # A matrix
		optimizer = torch.optim.Adam(self.A_matrix.parameters(), lr=self.lr, weight_decay=5e-4)
		input_is_latent = False
		recovered_step, self.A_matrix = self.utils_train.start_from_checkpoint(self.A_matrix, self.resume_training_model)
		if recovered_step != 0:
			print('Resume training from {}'.format(recovered_step))
		
		list_loss = []
		for step in range(recovered_step, self.n_steps, 1):

			loss_dict = {}
			self.G.zero_grad()
			source_z = make_noise(self.batch_size, self.dim_z, None).cuda()				
			target_z = make_noise(self.batch_size, self.dim_z, None).cuda()	
			
			with torch.no_grad():
				######## Source images ########
				imgs_source, source_w = generate_image(self.G, source_z, self.truncation, self.trunc, input_is_latent = input_is_latent, return_latents= True)
				params_source, angles_source = calculate_shapemodel(self.deca, imgs_source)
				
				######## Target images	 ########
				imgs_target = generate_image(self.G, target_z, self.truncation, self.trunc, input_is_latent= input_is_latent, return_latents = False)
				params_target, angles_target = calculate_shapemodel(self.deca, imgs_target)
				
			######## Generate Delta_p: difference of facial pose parameters ########
			if self.disentanglement_50:
				shift_vector, target_indices = self.utils_train.make_shift_vector_50(params_source, params_target, angles_source, angles_target)		
			else:
				target_indices = torch.zeros(1)
				shift_vector = self.utils_train.make_shift_vector(params_source, params_target, angles_source, angles_target)
			
			######## Predict shift in the latent space ########
			shift = self.A_matrix(shift_vector)
			######## Generate shifted image ########
			imgs_shifted, shifted_latents = generate_image(self.G, source_z, self.truncation, self.trunc, shift_code = shift, 
											input_is_latent= input_is_latent, return_latents=True)
			params_shifted, angles_shifted = calculate_shapemodel(self.deca, imgs_shifted)

			loss, loss_dict = self.utils_train.calculate_losses(params_source, angles_source, params_shifted, angles_shifted, params_target, angles_target, shift_vector, 
				target_indices, imgs_source, imgs_shifted)
			
			######## Total loss ########	
			list_loss.append(loss.data.item())		
			
			self.A_matrix.zero_grad()
			loss.backward()
			optimizer.step()

			######## Evaluate/Logs ########
			self.utils_train.log(self.G, self.A_matrix, step, loss_dict, np.mean(np.array(list_loss)), self.models_dir)
			if step % 500 == 0 and step > 0:
				list_loss = []
			if self.use_wandb:
				wandb.log({
					'step': step,
				})
				wandb.log(loss_dict)

	def train_real(self):

		if self.train_dataset_path is None or self.test_dataset_path is None:
			print('Specify the path for the training and validation dataset.')
			exit()

		self.load_models()
		if self.use_wandb:
			self.initialize_wandb()
		self.initialize_utilities_train()

		self.train_dataset, self.test_dataset = self.utils_train.configure_dataset()
		self.A_matrix.train() # A
		optimizer = torch.optim.Adam(self.A_matrix.parameters(), lr=self.lr, weight_decay=5e-4)
		input_is_latent = True
		list_loss = []
		epochs = self.n_steps
		
		if self.training_method == 'real':
			batch_local = self.batch_size
		else:
			assert (self.batch_size % 2 == 0), 'batch size should be an even number.'
			batch_local = int(self.batch_size/2)

		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=batch_local,
										   shuffle=True,
										   num_workers=int(self.workers),
										   drop_last=True)
		

		print('')
		print('******************* Dataset ********************')
		train_images, train_ids, train_videos = self.train_dataset.get_length()
		print('Training: {} images {} ids {} videos'.format(train_images, train_ids, train_videos))
		test_images = self.test_dataset.get_length()
		print('Validation: {} images'.format(test_images))
		print('************************************************')
		print('')
		print('Start training for {} epoch'.format(epochs))

		recovered_step, self.A_matrix = self.utils_train.start_from_checkpoint(self.A_matrix, self.resume_training_model)
		self.global_step = recovered_step
		if recovered_step != 0:
			print('Resume training from {}'.format(recovered_step))		
		self.epoch = 0			
		while self.epoch < epochs:
			for batch_idx, batch in enumerate(self.train_dataloader):
				
				loss_dict = {}
				self.G.zero_grad()	
				
				################## Source ##################
				sample_batch = batch
				source_w = sample_batch['w'].cuda()
				source_real_img = sample_batch['real_img'].cuda()
				source_inv_img = sample_batch['inv_img'].cuda()	
				# If both real and synthetic images sample synthetic	
				if self.training_method == 'real_synthetic':
					source_w = sample_batch['w'].cuda()
					source_z_syn = make_noise(batch_local, self.dim_z, None).cuda()	
					source_w_syn = self.G.get_latent(source_z_syn)
					source_w_syn = source_w_syn.unsqueeze(1).repeat(1, self.G.n_latent, 1)
					source_w = torch.cat((source_w, source_w_syn), dim = 0)
					imgs_source = generate_image(self.G, source_w_syn, self.truncation, self.trunc, input_is_latent = True, return_latents = False)
					source_real_img = torch.cat((source_real_img, imgs_source), dim = 0)		
				with torch.no_grad():
					params_source, angles_source  = calculate_shapemodel(self.deca, source_real_img)

				
				################## Target ##################
				target_w = make_noise(self.batch_size, self.dim_z, None).cuda()	
				with torch.no_grad():
					imgs_target = generate_image(self.G, target_w, self.truncation, self.trunc, input_is_latent = False, return_latents = False)
					params_target, angles_target  = calculate_shapemodel(self.deca, imgs_target)
				
				# Generate shift		
				if self.disentanglement_50:
					shift_vector, target_indices = self.utils_train.make_shift_vector_50(params_source, params_target, angles_source, angles_target)		
				else:
					target_indices = torch.zeros(1)
					shift_vector = self.utils_train.make_shift_vector(params_source, params_target, angles_source, angles_target)

				shift = self.A_matrix(shift_vector)
				imgs_shifted, shifted_latents = generate_image(self.G, source_w, self.truncation, self.trunc, shift_code = shift, input_is_latent= input_is_latent, return_latents=True)	
				params_shifted, angles_shifted = calculate_shapemodel(self.deca, imgs_shifted)
				
				loss, loss_dict = self.utils_train.calculate_losses(params_source, angles_source, params_shifted, angles_shifted, params_target, angles_target, shift_vector, 
					target_indices, source_real_img, imgs_shifted)
			
				############## Total loss ##############	
				list_loss.append(loss.data.item())
				
				self.A_matrix.zero_grad()
				loss.backward()
				optimizer.step()

				######### Evaluate #########
				self.utils_train.log(self.G, self.A_matrix, self.global_step, loss_dict, np.mean(np.array(list_loss)), self.models_dir)
				if self.global_step % 500 == 0 and self.global_step > 0:
					list_loss = []
				if self.use_wandb:
					wandb.log({
						'step': self.global_step,
					})
					wandb.log(loss_dict)

				self.global_step += 1

			self.epoch += 1
			
	def train_paired(self):
		if self.train_dataset_path is None or self.test_dataset_path is None:
			print('Specify the path for the training and validation dataset.')
			exit()

		self.load_models()
		if self.use_wandb:
			self.initialize_wandb()
		self.initialize_utilities_train()

		
		self.train_dataset, self.test_dataset = self.utils_train.configure_dataset()
		self.A_matrix.train() # A
		optimizer = torch.optim.Adam(self.A_matrix.parameters(), lr=self.lr, weight_decay=5e-4)
		input_is_latent = True
		list_loss = []
		epochs = self.n_steps

		print('')
		print('******************* Dataset ********************')
		train_images, train_ids, train_videos = self.train_dataset.get_length()
		print('Training: {} images {} ids {} videos'.format(train_images, train_ids, train_videos))
		print('Validation: {} images'.format(self.validation_samples))
		print('************************************************')
		print('')

		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.batch_size,
										   shuffle=True,
										   num_workers=int(self.workers),
										   drop_last=True)
		
		recovered_step, self.A_matrix = self.utils_train.start_from_checkpoint(self.A_matrix, self.resume_training_model)
		self.global_step = recovered_step
		if recovered_step != 0:
			print('Resume training from {}'.format(recovered_step))
		self.epoch = 0
		while self.epoch < epochs:
			for batch_idx, batch in enumerate(self.train_dataloader):
				
				loss_dict = {}
				self.G.zero_grad()	
				
				################## Source ##################
				sample_batch = batch
				source_w = sample_batch['source_latent_code'].cuda()
				source_img = sample_batch['source_img'].cuda()
				with torch.no_grad():
					params_source, angles_source  = calculate_shapemodel(self.deca, source_img)

				################## Target ##################
				target_w = sample_batch['target_latent_code'].cuda()
				target_img = sample_batch['target_img'].cuda()
				with torch.no_grad():
					params_target, angles_target  = calculate_shapemodel(self.deca, target_img)
				
				# Generate shift
				target_indices = np.zeros(self.batch_size)
				shift_vector = self.utils_train.make_shift_vector(params_source, params_target, angles_source, angles_target)
				shift = self.A_matrix(shift_vector)
				imgs_shifted, shifted_latents = generate_image(self.G, source_w, self.truncation, self.trunc, shift_code = shift, 
																					input_is_latent= input_is_latent, return_latents=True)
					
				params_shifted, angles_shifted = calculate_shapemodel(self.deca, imgs_shifted)

				loss, loss_dict = self.utils_train.calculate_losses_paired(params_shifted, params_target, imgs_shifted, target_img, shifted_latents, target_w)
				############## Total loss ##############	
				list_loss.append(loss.data.item())
				
				self.A_matrix.zero_grad()
				loss.backward()
				optimizer.step()

				######### Evaluate #########
				self.utils_train.log(self.G, self.A_matrix, self.global_step, loss_dict, np.mean(np.array(list_loss)), self.models_dir, epoch = self.epoch)
				if self.global_step % 500 == 0 and self.global_step > 0:
					list_loss = []
				if self.use_wandb:
					wandb.log({
						'step': self.global_step,
					})
					wandb.log(loss_dict)

				self.global_step += 1

			self.epoch += 1
			# Init dataloader again. For each epoch load only 2 pairs from each video
			train_dataset = CustomDataset_paired(self.train_dataset_path, max_pairs = 2)
			self.train_dataloader = DataLoader(train_dataset,
										   batch_size=self.batch_size,
										   shuffle=True,
										   num_workers=int(self.workers),
										   drop_last=True)
			