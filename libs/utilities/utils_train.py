import numpy as np
import os
import wandb
import cv2
import imageio
import torch
from torch import nn
import torchvision 
from torchvision import utils as torch_utils
from torch.utils.data import DataLoader

from libs.utilities.image_utils import torch_range_1_to_255
from libs.criteria.losses import Losses
from libs.criteria import id_loss
from libs.criteria.lpips.lpips import LPIPS
from libs.configs.config_directions import *
from libs.utilities.generic import initialize_directions, calculate_shapemodel, save_image, generate_image
from libs.utilities.utils import make_noise
from libs.utilities.utils_inference import generate_grid_image
from libs.DECA.decalib.utils.rotation_converter import batch_euler2axis, deg2rad, rad2deg, batch_axis2euler
from libs.utilities.visualization import make_interpolation_chart
from libs.datasets.dataloader import CustomDataset, CustomDataset_testset_synthetic, CustomDataset_testset_real
from libs.datasets.dataloader_paired import CustomDataset_paired, CustomDataset_paired_validation
from libs.configs.config_directions import *

class Utilities_train(object):
	def __init__(self, params, shape_model, images_dir, images_reenact_dir, truncation, trunc, wandb):
		self.deca = shape_model
		self.params = params
		self.images_dir = images_dir
		self.images_reenact_dir = images_reenact_dir
		self.truncation = truncation
		self.trunc = trunc
		self.wandb = wandb
		self.shift_scale = self.params['shift_scale']
		if self.params['dataset_type'] == 'voxceleb':
			directions_dict = voxceleb_dict
		if self.params['dataset_type'] == 'ffhq':
			directions_dict = ffhq_dict
		self.yaw_direction = directions_dict['yaw_direction']
		self.pitch_direction = directions_dict['pitch_direction']
		self.roll_direction = directions_dict['roll_direction']
		self.batch_size = self.params['batch_size']
		self.steps_per_log = self.params['steps_per_log'] 
		self.steps_per_save = self.params['steps_per_save'] 
		self.steps_per_ev_log = self.params['steps_per_ev_log'] 
		self.reenactment_fig = self.params['reenactment_fig']
		self.gif = self.params['gif']
		self.learned_directions = self.params['learned_directions']
		self.evaluation = self.params['evaluation']

		##### Load pretrained models #####
		self.id_loss_ = id_loss.IDLoss().cuda().eval()
		self.lpips_loss = LPIPS(net_type='alex').cuda().eval()
		self.losses = Losses()

		self.count_pose, self.num_expressions, self.directions_exp, jaw_dict, self.angle_scales, self.angle_directions = initialize_directions(self.params['dataset_type'], 
			self.params['learned_directions'], self.params['shift_scale'])

		self.a_jaw = jaw_dict['a']
		self.b_jaw = jaw_dict['b']
		self.max_jaw = jaw_dict['max']
		self.min_jaw = jaw_dict['min']

	def configure_dataset(self):
		self.out_dir = self.params['experiment_path']
		
		if self.params['training_method'] == 'synthetic':
			
			self.test_dataset = CustomDataset_testset_synthetic(synthetic_dataset_path = self.params['synthetic_dataset_path'], num_samples = self.params['validation_samples'])	
			self.test_batch_size = self.params['test_batch_size']
			if self.test_batch_size > self.test_dataset.num_samples:
				self.test_batch_size = self.test_dataset.num_samples
			self.test_dataloader = DataLoader(self.test_dataset,
										batch_size=self.test_batch_size ,
										shuffle=False,
										num_workers=int(self.params['workers']),
										drop_last=True)	
						
			if self.gif:
				self.z_fixed = self.test_dataset.fixed_source_w[0].unsqueeze(0).cuda()
			
		if self.params['training_method'] == 'real' or self.params['training_method'] == 'real_synthetic':

			#### Training dataset ####
			self.train_dataset = CustomDataset(self.params['train_dataset_path'])	
			
			#### Validation dataset ####
			self.test_dataset = CustomDataset_testset_real(dataset_path = self.params['test_dataset_path'], num_samples = self.params['validation_samples'])

			
			self.test_batch_size = self.params['test_batch_size']
			if self.test_batch_size > self.test_dataset.num_samples:
				self.test_batch_size = self.test_dataset.num_samples

			self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.params['workers']),
										  drop_last=True)
	
			if self.gif:
				source_w = torch.from_numpy(np.load(self.test_dataset.w[0]))
				self.z_fixed = source_w.unsqueeze(0).cuda()

			return self.train_dataset, self.test_dataset
				
		if self.params['training_method'] == 'paired':
			self.train_dataset = CustomDataset_paired(self.params['train_dataset_path'], max_pairs = 2)

			self.test_dataset = CustomDataset_paired_validation(self.params['test_dataset_path'],  num_samples = self.params['validation_samples'])
			
			
			self.test_batch_size = self.params['test_batch_size']
			if self.test_batch_size > self.test_dataset.num_samples:
				self.test_batch_size = self.test_dataset.num_samples

			shuffled_dataset = torch.utils.data.Subset(self.test_dataset, torch.randperm(len(self.test_dataset)).tolist())
			self.test_dataloader = DataLoader(shuffled_dataset,
								batch_size=self.test_batch_size ,
								shuffle=False,
								num_workers=int(self.params['workers']),
								drop_last=True)		

			return self.train_dataset, self.test_dataset
		
	def make_shift_vector(self, param_source, param_target, angles_source, angles_target):
		
		
		shift_vector = torch.zeros(self.params['batch_size'], self.params['learned_directions']).cuda()
		
		if self.yaw_direction != -1:
			angles_source_shift_sc = angles_source[:, 0] * self.shift_scale / self.angle_scales[0]
			angles_target_shift_sc = angles_target[:, 0] * self.shift_scale / self.angle_scales[0]
			yaw_shift =  angles_target_shift_sc - angles_source_shift_sc
			shift_vector[:, self.yaw_direction] = yaw_shift

		if self.pitch_direction != -1:
			angles_source_shift_sc = angles_source[:, 1] * self.shift_scale / self.angle_scales[1]
			angles_target_shift_sc = angles_target[:, 1] * self.shift_scale / self.angle_scales[1]
			pitch_shift =  angles_target_shift_sc - angles_source_shift_sc
			shift_vector[:, self.pitch_direction] = pitch_shift
		
		if self.roll_direction != -1:
			angles_source_shift_sc = angles_source[:, 2] * self.shift_scale / self.angle_scales[2]
			angles_target_shift_sc = angles_target[:, 2] * self.shift_scale / self.angle_scales[2]
			roll_shift =  angles_target_shift_sc - angles_source_shift_sc
			shift_vector[:, self.roll_direction] = roll_shift

		jaw_exp_source = param_source['pose'][:, 3]
		jaw_exp_target = param_target['pose'][:, 3]
		a = self.a_jaw
		b = self.b_jaw
		target = a * jaw_exp_target + b
		source = a * jaw_exp_source + b
		shift_exp = target - source
		shift_vector[:, self.count_pose - 1] = shift_exp
	
		exp_target = param_target['alpha_exp']
		exp_source = param_source['alpha_exp']
		for index in range(self.num_expressions):		
			ind_exp = self.directions_exp[index]['exp_component']
			target_expression = exp_target[:, ind_exp]
			
			source_expression = exp_source[:, ind_exp]
			
			a = self.directions_exp[index]['a']
			b = self.directions_exp[index]['b']	
			target = a * target_expression + b
			source = a * source_expression + b
			shift_exp = target - source	
			shift_vector[:, index + self.count_pose ] = shift_exp

		return shift_vector

	" Half first of batch size full reenactment, half second of batch size only one direction! "
	def make_shift_vector_50(self, param_source, param_target, angles_source, angles_target):
		
		if self.params['batch_size'] % 2 !=0:
			print('Error. Batch size should be even number!')
			exit()

		shift_vector = torch.zeros(self.params['batch_size'], self.params['learned_directions']).cuda()
		batch_2 = int(self.params['batch_size']/2)
		if self.yaw_direction != -1:
			angles_source_shift_sc = angles_source[:batch_2, 0] * self.shift_scale / self.angle_scales[0]
			angles_target_shift_sc = angles_target[:batch_2, 0] * self.shift_scale / self.angle_scales[0]
			yaw_shift =  angles_target_shift_sc - angles_source_shift_sc
			shift_vector[:batch_2, self.yaw_direction] = yaw_shift

		if self.pitch_direction != -1:
			angles_source_shift_sc = angles_source[:batch_2, 1] * self.shift_scale / self.angle_scales[1]
			angles_target_shift_sc = angles_target[:batch_2, 1] * self.shift_scale / self.angle_scales[1]
			pitch_shift =  angles_target_shift_sc - angles_source_shift_sc
			shift_vector[:batch_2, self.pitch_direction] = pitch_shift
		
		if self.roll_direction != -1:
			angles_source_shift_sc = angles_source[:batch_2, 2] * self.shift_scale / self.angle_scales[2]
			angles_target_shift_sc = angles_target[:batch_2, 2] * self.shift_scale / self.angle_scales[2]
			roll_shift =  angles_target_shift_sc - angles_source_shift_sc
			shift_vector[:batch_2, self.roll_direction] = roll_shift

		jaw_exp_source = param_source['pose'][:batch_2, 3]
		jaw_exp_target = param_target['pose'][:batch_2, 3]
		a = self.a_jaw
		b = self.b_jaw
		target = a * jaw_exp_target + b
		source = a * jaw_exp_source + b
		shift_exp = target - source
		shift_vector[:batch_2, self.count_pose - 1] = shift_exp
			
		exp_target = param_target['alpha_exp']
		exp_source = param_source['alpha_exp']
		for index in range(self.num_expressions):		
			ind_exp = self.directions_exp[index]['exp_component']
			target_expression = exp_target[:batch_2, ind_exp]
			source_expression = exp_source[:batch_2, ind_exp]
			
			a = self.directions_exp[index]['a']
			b = self.directions_exp[index]['b']	
			target = a * target_expression + b
			source = a * source_expression + b
			shift_exp = target - source
			
			shift_vector[:batch_2, index + self.count_pose ] = shift_exp


		dir_array = list(range(0, self.params['learned_directions']))
		dir_array = np.asarray(dir_array)
		num_directions = list(range(0, self.params['learned_directions']))
		num_directions = np.asarray(num_directions)
		target_indices =  np.random.choice(num_directions , size = int(self.params['batch_size']/2)) 
		count = 0
		for batch in range(int(self.params['batch_size']/2), self.params['batch_size']):
			ind  = int(target_indices[count])
			if ind == self.yaw_direction:
				start_pose = angles_source[batch, 0].clone() * self.shift_scale / self.angle_scales[0]
				min_shift = (-self.shift_scale - start_pose ).cuda()
				max_shift = (self.shift_scale - start_pose).cuda()
				shift  = (min_shift - max_shift) * torch.rand(1, device='cuda') + max_shift
				shift_vector[batch, ind] = shift
				count += 1
				continue
			if ind == self.pitch_direction:
				start_pose = angles_source[batch, 1] * self.shift_scale / self.angle_scales[1]
				min_shift = (-self.shift_scale - start_pose).cuda()
				max_shift = (self.shift_scale - start_pose).cuda()
				shift  = (min_shift - max_shift) * torch.rand(1, device='cuda') + max_shift
				shift_vector[batch, ind] = shift
				count += 1
				continue
		
			if ind == self.roll_direction:
				start_pose = angles_source[batch, 2] * self.shift_scale / self.angle_scales[2]
				min_shift = (-self.shift_scale - start_pose).cuda()
				max_shift = (self.shift_scale - start_pose).cuda()
				shift  = (min_shift - max_shift) * torch.rand(1, device='cuda') + max_shift
				shift_vector[batch, ind] = shift
				count += 1
				continue
		
			if ind == self.count_pose-1:
				jaw_exp_source = param_source['pose'][batch, 3]
				a = self.a_jaw
				b = self.b_jaw
				start_pose = a * jaw_exp_source + b			
				min_shift = (-self.shift_scale - start_pose).cuda()
				max_shift = (self.shift_scale - start_pose).cuda()
				shift  = (min_shift - max_shift) * torch.rand(1, device='cuda') + max_shift
				shift_vector[batch, ind] = shift
				count += 1
				continue
				
			index = next((index for (index, d) in enumerate(self.directions_exp) if d['A_direction'] == ind), None)
			if index is not None:
				ind_exp = self.directions_exp[index]['exp_component']
				exp_source = param_source['alpha_exp']
				exp_1 = exp_source[batch][ind_exp].clone()
				a = self.directions_exp[index]['a']
				b = self.directions_exp[index]['b']
				start = a * exp_1 + b			
				min_shift = (-self.shift_scale - start)
				max_shift = (self.shift_scale - start)
				shift  = (min_shift - max_shift) * torch.rand(1, device='cuda') + max_shift
				shift_vector[batch, ind] = shift
				count += 1
	
		return shift_vector, target_indices

	" Generate ground truth reenacted shape "
	def get_params_gt_reenacted(self, param_source, param_target, shift_vector, target_indices, angles_source):

		coefficients_gt = {}

		coefficients_gt['pose'] = param_source['pose'].clone()
		coefficients_gt['exp'] = param_source['alpha_exp'].clone()
		for batch in range(int(self.batch_size/2)):
			coefficients_gt['pose'][batch] = param_target['pose'][batch]
			coefficients_gt['exp'][batch] = param_target['alpha_exp'][batch]

		count = 0
		for batch in range(int(self.batch_size/2), self.batch_size):
			ind  = int(target_indices[count])
			if ind == self.yaw_direction:
				start_pose = angles_source[batch, 0].clone() * self.shift_scale / self.angle_scales[0]
				shift = shift_vector[batch, ind]
				target_yaw = (start_pose + shift) * self.angle_scales[0]/self.shift_scale
				angles_tmp = angles_source[batch].clone()
				angles_tmp[0] = target_yaw
				target_yaw_pose = batch_euler2axis(deg2rad(angles_tmp.unsqueeze(0)))[0]
				pose_tmp_2 = target_yaw_pose.clone()
				target_yaw_pose[0] = pose_tmp_2[1]
				target_yaw_pose[1] = -pose_tmp_2[0]
				coefficients_gt['pose'][batch][:3] = target_yaw_pose
				pose = rad2deg(batch_axis2euler(target_yaw_pose.unsqueeze(0)))
		
				count += 1
				continue
			if ind == self.pitch_direction:
				start_pose = angles_source[batch, 1].clone() * self.shift_scale / self.angle_scales[1]
				shift = shift_vector[batch, ind]
				target_pitch = (start_pose + shift) * self.angle_scales[1]/self.shift_scale
				angles_tmp = angles_source[batch].clone()
				angles_tmp[1] = target_pitch
				target_pitch_pose = batch_euler2axis(deg2rad(angles_tmp))[0]	
				pose_tmp_2 = target_pitch_pose.clone()
				target_pitch_pose[0] = pose_tmp_2[1]
				target_pitch_pose[1] = -pose_tmp_2[0]
				coefficients_gt['pose'][batch][:3] = target_pitch_pose

				count += 1
				continue
			if ind == self.roll_direction:
				start_pose = angles_source[batch, 2] * self.shift_scale / self.angle_scales[2]
				shift = shift_vector[batch, ind]
				target_roll = (start_pose + shift) * self.angle_scales[2]/self.shift_scale

				angles_tmp = angles_source[batch].clone()
				angles_tmp[2] = target_roll
				target_roll_pose = batch_euler2axis(deg2rad(angles_tmp))[0]	
				pose_tmp_2 = target_roll_pose.clone()
				target_roll_pose[0] = pose_tmp_2[1]
				target_roll_pose[1] = -pose_tmp_2[0]
				coefficients_gt['pose'][batch][:3] = target_roll_pose

				count += 1
				continue

			if ind == self.count_pose -1: # Jaw direction
				jaw_exp_source = param_source['pose'][batch, 3]
				a = self.a_jaw
				b = self.b_jaw
				shift = shift_vector[batch, ind]
				start_pose = a * jaw_exp_source + b
				target_jaw =  ((start_pose + shift) - b)/a
				coefficients_gt['pose'][batch][3] = target_jaw
				count += 1
				continue
			
			index = next((index for (index, d) in enumerate(self.directions_exp) if d['A_direction'] == ind), None)
			if index is not None:
				ind_exp = self.directions_exp[index]['exp_component']
				shift = shift_vector[batch, ind]
				exp_source = param_source['alpha_exp']
				exp_1 = exp_source[batch][ind_exp].clone()
				a = self.directions_exp[index]['a']
				b = self.directions_exp[index]['b']
				start_pose = a * exp_1 + b
				target_exp =  ((start_pose + shift) - b)/a
				coefficients_gt['exp'][batch][ind_exp] = target_exp

				count += 1

		return coefficients_gt

	def calculate_losses(self, params_source, angles_source, params_shifted, angles_shifted, params_target, angles_target, shift_vector, target_indices,
			imgs_source, imgs_shifted):

		loss_dict = {} 
		loss = 0
		
		############## Shape Loss ##############
		if self.params['lambda_shape'] > 0:
			coefficients_gt = {}
			############# Ground truth reenacted image: Source identity - Target pose #############
			# If disentanglement_50 = True then the ground truth reenacted face has only one facial attribute (yaw, pitch, roll, smile etc. from the target face)		
			if self.params['disentanglement_50']:
				coefficients_gt = self.get_params_gt_reenacted( params_source, params_target, shift_vector, target_indices, angles_source)
			else:
				coefficients_gt['pose'] = params_target['pose']
				coefficients_gt['exp'] = params_target['alpha_exp']
			coefficients_gt['cam'] = params_shifted['cam']
			coefficients_gt['cam'][:,:] = 0.
			coefficients_gt['cam'][:,0] = 8
			coefficients_gt['shape'] = params_source['alpha_shp']
			landmarks2d_gt, landmarks3d_gt, shape_gt = self.deca.calculate_shape(coefficients_gt)
			###########################################################################################

			####################################### Reenacted image ###################################
			coefficients_reen = {}
			coefficients_reen['pose'] = params_shifted['pose']
			coefficients_reen['shape'] = params_shifted['alpha_shp']
			coefficients_reen['exp'] = params_shifted['alpha_exp']
			coefficients_reen['cam'] = params_shifted['cam']
			coefficients_reen['cam'][:,:] = 0.
			coefficients_reen['cam'][:,0] = 8
			landmarks2d_reenacted, landmarks3d_reenacted, shape_reenacted = self.deca.calculate_shape(coefficients_reen)
			###########################################################################################

			loss_shape = self.params['lambda_shape'] *  self.losses.calculate_shape_loss(shape_gt, shape_reenacted, normalize = False)
			loss_mouth = self.params['lambda_mouth_shape'] *  self.losses.calculate_mouth_loss(landmarks2d_gt, landmarks2d_reenacted) 
			loss_eye = self.params['lambda_eye_shape'] * self.losses.calculate_eye_loss(landmarks2d_gt, landmarks2d_reenacted)
			loss_dict['loss_shape'] = loss_shape.data.item()
			loss_dict['loss_eye'] = loss_eye.data.item()
			loss_dict['loss_mouth'] = loss_mouth.data.item()

			loss += loss_mouth
			loss += loss_shape
			loss += loss_eye
			
		############## Identity losses ##############	
		if self.params['lambda_identity'] != 0:		
			loss_identity = self.params['lambda_identity'] * self.id_loss_(imgs_shifted, imgs_source.detach())
			loss_dict['loss_identity'] = loss_identity.data.item()
			loss += loss_identity

		if self.params['lambda_perceptual'] != 0:
			loss_perceptual = self.params['lambda_perceptual'] * self.lpips_loss(imgs_shifted, imgs_source.detach())
			loss_dict['loss_perceptual'] = loss_perceptual.data.item()
			loss += loss_perceptual

		loss_dict['loss'] = loss.data.item()
		return loss, loss_dict
	
	def calculate_losses_paired(self, params_shifted, params_target, imgs_shifted, imgs_target, shifted_latents, target_w):
		loss_dict = {} 
		loss = 0
		imgs_shifted_255 = torch_range_1_to_255(imgs_shifted)
		imgs_target_255 = torch_range_1_to_255(imgs_target)
		############## Shape Loss ##############
		if self.params['lambda_shape'] > 0:
			coefficients_gt = {}
			
			coefficients_gt['pose'] = params_target['pose']
			coefficients_gt['exp'] = params_target['alpha_exp']
			coefficients_gt['cam'] = params_target['cam']
			coefficients_gt['cam'][:,:] = 0.
			coefficients_gt['cam'][:,0] = 8
			coefficients_gt['shape'] = params_target['alpha_shp']
			landmarks2d_gt, landmarks3d_gt, shape_gt = self.deca.calculate_shape(coefficients_gt)
			###########################################################################################

			####################################### Reenacted image ###################################
			coefficients_reen = {}
			coefficients_reen['pose'] = params_shifted['pose']
			coefficients_reen['shape'] = params_shifted['alpha_shp']
			coefficients_reen['exp'] = params_shifted['alpha_exp']
			coefficients_reen['cam'] = params_shifted['cam']
			coefficients_reen['cam'][:,:] = 0.
			coefficients_reen['cam'][:,0] = 8
			landmarks2d_reenacted, landmarks3d_reenacted, shape_reenacted = self.deca.calculate_shape(coefficients_reen)
			###########################################################################################

			loss_shape = self.params['lambda_shape'] *  self.losses.calculate_shape_loss(shape_gt, shape_reenacted, normalize = False)
			loss_mouth = self.params['lambda_mouth_shape'] *  self.losses.calculate_mouth_loss(landmarks2d_gt, landmarks2d_reenacted) 
			loss_eye = self.params['lambda_eye_shape'] * self.losses.calculate_eye_loss(landmarks2d_gt, landmarks2d_reenacted)
			loss_dict['loss_shape'] = loss_shape.data.item()
			loss_dict['loss_eye'] = loss_eye.data.item()
			loss_dict['loss_mouth'] = loss_mouth.data.item()

			loss += loss_mouth
			loss += loss_shape
			loss += loss_eye
			
		############## Identity losses ##############	
		if self.params['lambda_identity'] != 0:		
			loss_identity = self.params['lambda_identity'] * self.id_loss_(imgs_shifted, imgs_target.detach())
			loss_dict['loss_identity'] = loss_identity.data.item()
			loss += loss_identity

		if self.params['lambda_perceptual'] != 0:
			
			loss_perceptual = self.params['lambda_perceptual'] * self.lpips_loss(imgs_shifted_255, imgs_target_255.detach())
			loss_dict['loss_perceptual'] = loss_perceptual.data.item()
			loss += loss_perceptual

		if self.params['lambda_pixel_wise'] != 0:
			loss_pixel_wise = self.params['lambda_pixel_wise'] * self.losses.calculate_pixel_wise_loss(imgs_shifted_255, imgs_target_255.detach())
			loss_dict['loss_pixel_wise'] = loss_pixel_wise.data.item()
			loss += loss_pixel_wise

		if self.params['lambda_w_reg'] != 0:
			self.criterion_l1 = torch.nn.L1Loss()
			loss_w_reg = self.params['lambda_w_reg'] * self.criterion_l1(shifted_latents, target_w)
			loss_dict['loss_w_reg'] = loss_w_reg.data.item()
			loss += loss_w_reg

		loss_dict['loss'] = loss.data.item()
		return loss, loss_dict			
	
	def log(self, G, A_matrix, step, loss_dict, mean_loss, models_dir, epoch = None):
		# Print loss values every steps_per_log iterations = 10
		if step % self.steps_per_log == 0:
			self.log_train(step, mean_loss, loss_dict, epoch = epoch)

		if step % self.steps_per_ev_log == 0 and self.evaluation:	
			if self.params['training_method'] == 'paired':
				self.evaluate_model_reenactment_video(G, A_matrix, step)
			else:
				self.log_interpolation(G, A_matrix, step)
		
		# Save A_matrix model every steps_per_save iterations
		if step % self.steps_per_save == 0 and step > 0:
			self.save_models(A_matrix, step, models_dir)
	
	' Print loss values '
	def log_train(self, step, mean_loss, loss_dict,  epoch = None):
		
		if epoch is not None:
			out_text = '[epoch {:04d}, step {}]'.format(epoch, step)
		else:
			out_text = '[step {}]'.format(step)
		
		for key, value in loss_dict.items():
			out_text += (' | {}: {:.2f}'.format(key, value))
		out_text += '| Mean Loss {:.2f}'.format(mean_loss)
		print(out_text)
			
	'First function, random target indices'
	def make_shifts_interpolation(self, param_source, param_target, angles_source, angles_target):
		shift_vector_1 = torch.zeros(self.test_batch_size, self.learned_directions).cuda()

		if self.yaw_direction != -1:	
			angles_source_shift_sc = angles_source[:, 0] * self.shift_scale / self.angle_scales[0]
			angles_target_shift_sc = angles_target[:, 0] * self.shift_scale / self.angle_scales[0]
			yaw_shift =  angles_target_shift_sc - angles_source_shift_sc
			shift_vector_1[:, self.yaw_direction] = yaw_shift

		if self.pitch_direction != -1:
			angles_source_shift_sc = angles_source[:, 1] * self.shift_scale / self.angle_scales[1]
			angles_target_shift_sc = angles_target[:, 1] * self.shift_scale / self.angle_scales[1]
			pitch_shift =  angles_target_shift_sc - angles_source_shift_sc
			shift_vector_1[:, self.pitch_direction] = pitch_shift
		
		if self.roll_direction != -1:
			angles_source_shift_sc = angles_source[:, 2] * self.shift_scale / self.angle_scales[2]
			angles_target_shift_sc = angles_target[:, 2] * self.shift_scale / self.angle_scales[2]
			roll_shift =  angles_target_shift_sc - angles_source_shift_sc
			shift_vector_1[:, self.roll_direction] = roll_shift

		jaw_exp_source = param_source['pose'][:, 3]
		jaw_exp_target = param_target['pose'][:, 3]
		a = self.a_jaw
		b = self.b_jaw
		target = a * jaw_exp_target + b
		source = a * jaw_exp_source + b
		shift_exp = target - source
		shift_vector_1[:, self.count_pose - 1] = shift_exp
	
		exp_target = param_target['alpha_exp']
		exp_source = param_source['alpha_exp']
		for index in range(self.num_expressions):		
			ind_exp = self.directions_exp[index]['exp_component']
			target_expression = exp_target[:, ind_exp]
			
			source_expression = exp_source[:, ind_exp]
			
			a = self.directions_exp[index]['a']
			b = self.directions_exp[index]['b']	
			target = a * target_expression + b
			source = a * source_expression + b
			shift_exp = target - source	
			shift_vector_1[:, index + self.count_pose ] = shift_exp

		return shift_vector_1

	'Start training from model'
	def start_from_checkpoint(self, A_matrix, resume_training_model):
		
		step = 0
		if resume_training_model is not None:
			if os.path.isfile(resume_training_model):
				print('Resuming training from {}'.format(resume_training_model))
				state_dict = torch.load(resume_training_model)
				if step in state_dict:
					step = state_dict['step']
				A_matrix.load_state_dict(state_dict['A_matrix'])
				
		return step, A_matrix

	'Save models'
	def save_models(self, A_matrix, step, models_dir):
		
		state_dict = {
			'step': 					step,
			'A_matrix': 				A_matrix.state_dict(),
			'learned_directions':		self.learned_directions,
			'shift_scale':				self.shift_scale,
			'w_plus':					self.params['w_plus']	,
			'num_layers_shift':			self.params['num_layers_shift'],
		}
		checkpoint_path =  os.path.join(models_dir, 'A_matrix_{:06d}.pt'.format(step))
		torch.save(state_dict, checkpoint_path)
	
	def save_images(self, images, step, type_im, save_images_dir):
			
		grid = torch_utils.save_image(
			images,
			os.path.join(save_images_dir, '{:04d}_{}.png'.format(step, type_im)),
			normalize=True,
			range=(-1, 1),

		)
	
	def save_reenactment(self, source, target, reenacted, step, batch_size, save_images_dir, save = True, prefix = None):
		image_resolution = source[0].shape[1]
		grid_image = torch.zeros((3, batch_size*image_resolution, 3*image_resolution))
		for i in range(source.shape[0]):
			grid_image[:, i*image_resolution: (i+1)*image_resolution, :image_resolution] = source[i]
			grid_image[:, i*image_resolution: (i+1)*image_resolution, image_resolution:image_resolution*2] = target[i]
			grid_image[:, i*image_resolution: (i+1)*image_resolution, image_resolution*2:] = reenacted[i]

		if grid_image.shape[2] > 3* 256:
			width = 3* 256
			w = grid_image.shape[2]
			h = grid_image.shape[1]
			r = width / float(w)
			dim = (width, int(h * r))

			grid_image = torch.nn.functional.interpolate(grid_image,size=dim, mode='bilinear')

		if prefix is None:
			save_path = os.path.join(save_images_dir, '{:04d}.png'.format(step))
		else:
			save_path = os.path.join(save_images_dir, '{:04d}_{}.png'.format(step, prefix))
			
		if save:
			grid = torch_utils.save_image(
				grid_image,
				save_path,
				normalize=True,
				range=(-1, 1),
				nrow=0,
			)
		else:
			return grid
	
	##################################################################################
	################################### Evaluation ###################################
	##################################################################################

	'Generate interpolation chart saved in logs/images'
	def log_interpolation(self, G, A_matrix, step):
		
		A_matrix.eval()		
		self.evaluate_model_reenactment(G, A_matrix, step)

		# Interpolate on each direction.
		if self.gif:
			z = self.z_fixed
			if self.params['training_method'] == 'real' or self.params['training_method'] == 'real_synthetic':
				input_is_latent = True
			else:
				input_is_latent = False
			
			learned_directions = self.params['learned_directions']		
			info ={
				'shift_scale':  				self.params['shift_scale'],
				'shifts_count':				4,
				'generate_image':			True,
				'w_plus': 					self.params['w_plus'],
				'input_is_latent':			input_is_latent,
				'learned_dims':				learned_directions,
				'num_layers_shift':			self.params['num_layers_shift'],
				'count_pose':				self.count_pose,
				'a_jaw':					self.a_jaw,
				'b_jaw':					self.b_jaw,
				'max_jaw':					self.max_jaw,
				'min_jaw':					self.min_jaw,
			}
			
			max_directions = 5 # show only the first 5 directions
			if learned_directions < max_directions:
				max_directions = learned_directions

			grids, types  = make_interpolation_chart(G, A_matrix, self.deca,
				z, self.directions_exp, self.angle_scales, self.angle_directions, info,max_directions = max_directions)
		
			for i in range(len(grids)):
				fig_file_path = os.path.join(self.images_dir, 'gif_{:06d}_{}.gif'.format(step, types[i]))	
				imageio.mimsave(fig_file_path, grids[i])
	
		A_matrix.train()

	def extract_evaluation_metrics(self, params_shifted, params_target, angles_shifted, angles_target, imgs_shifted, imgs_source):
		############ Evaluation ############
		yaw_reenacted = angles_shifted[:,0][0].detach().cpu().numpy() 
		pitch_reenacted = angles_shifted[:,1][0].detach().cpu().numpy() 
		roll_reenacted = angles_shifted[:,2][0].detach().cpu().numpy()
		exp_reenacted = params_shifted['alpha_exp'][0].detach().cpu().numpy() 
		jaw_reenacted = params_shifted['pose'][0, 3].detach().cpu().numpy() 
		
		yaw_target = angles_target[:,0][0].detach().cpu().numpy() 
		pitch_target = angles_target[:,1][0].detach().cpu().numpy() 
		roll_target = angles_target[:,2][0].detach().cpu().numpy()
		exp_target = params_target['alpha_exp'][0].detach().cpu().numpy() 
		jaw_target = params_target['pose'][0, 3].detach().cpu().numpy()

		## normalize exp coef in [0,1]
		exp_error = []		
		for j  in range(self.learned_directions - self.count_pose):
			max_range = self.directions_exp[j]['max_shift']
			min_range = self.directions_exp[j]['min_shift']
			gt_pred_norm = (exp_target[j] - min_range)/(max_range-min_range)
			shifted_pred_norm = (exp_reenacted[j] - min_range)/(max_range-min_range)
			exp_error.append(abs(shifted_pred_norm - gt_pred_norm))
		

		gt_pred_norm = (jaw_target - self.min_jaw)/(self.max_jaw-self.min_jaw)
		shifted_pred_norm = (jaw_reenacted - self.min_jaw)/(self.max_jaw-self.min_jaw)
		exp_error.append(abs(shifted_pred_norm - gt_pred_norm))
		exp_error = np.mean(exp_error)
		
		pose = abs(yaw_reenacted-yaw_target) + abs(pitch_reenacted-pitch_target) + abs(roll_reenacted-roll_target)
		pose = pose/3	
		################################################

		# CSIM
		loss_identity = self.id_loss_(imgs_shifted, imgs_source) 
		csim = 1 - loss_identity.data.item()

		return csim, pose, exp_error

	'Evaluate models for face reenactment and save reenactment figure'
	def evaluate_model_reenactment(self, G, A_matrix, step):
		if self.params['training_method'] == 'synthetic':
			input_is_latent = False
		elif self.params['training_method'] == 'real' or self.params['training_method'] == 'real_synthetic':
			input_is_latent = True
			
		exp_error = 0 ; pose_error = 0; csim_total = 0
		count = 0
		counter_logs = 0
		num_pairs_log = self.params['num_pairs_log']
		if self.reenactment_fig:
			source_images = torch.zeros((num_pairs_log, 3, 256, 256))
			target_images = torch.zeros((num_pairs_log, 3, 256, 256))
			reenacted_images = torch.zeros((num_pairs_log, 3, 256, 256))
		with torch.no_grad():
			for batch_idx, batch in enumerate(self.test_dataloader):
				sample_batch = batch

				source_w = sample_batch['source_w'].cuda()
				target_w = sample_batch['target_w'].cuda()
				
				imgs_source, w = generate_image(G, source_w, self.truncation, self.trunc, return_latents = True, input_is_latent = input_is_latent)
				params_source, angles_source = calculate_shapemodel(self.deca, imgs_source)
						
				imgs_target  = generate_image(G, target_w, self.truncation, self.trunc, return_latents = False, input_is_latent = False)	
				params_target, angles_target = calculate_shapemodel(self.deca, imgs_target)

				shift_vector = self.make_shifts_interpolation(params_source, params_target, angles_source, angles_target)	
				shift = A_matrix(shift_vector)
				imgs_shifted, shifted_latents = generate_image(G, source_w, self.truncation, self.trunc, 
					shift_code = shift, input_is_latent= input_is_latent, return_latents=True)
				params_shifted, angles_shifted = calculate_shapemodel(self.deca, imgs_shifted)
				
				csim, pose, exp = self.extract_evaluation_metrics(params_shifted, params_target, angles_shifted, angles_target, imgs_shifted, imgs_source)
				exp_error += exp
				csim_total += csim
				pose_error += pose
				count += 1

				if self.reenactment_fig:
					if counter_logs < num_pairs_log:
						if (num_pairs_log - counter_logs) % source_w.shape[0] == 0:
							source_images[counter_logs:counter_logs+source_w.shape[0]] = imgs_source.detach().cpu()
							target_images[counter_logs:counter_logs+source_w.shape[0]] = imgs_target.detach().cpu()
							reenacted_images[counter_logs:counter_logs+source_w.shape[0]] = imgs_shifted.detach().cpu()
						else:
							num = num_pairs_log - counter_logs
							source_images[counter_logs:counter_logs+num]  = imgs_source[:num].detach().cpu()
							target_images[counter_logs:counter_logs+num]  = imgs_target[:num].detach().cpu()
							reenacted_images[counter_logs:counter_logs+num]  = imgs_shifted[:num].detach().cpu()
						counter_logs += source_w.shape[0]

		if self.reenactment_fig:
			sample = generate_grid_image(source_images, target_images, reenacted_images)
			save_image(sample, os.path.join(self.images_reenact_dir, '{:06d}.png'.format(step)))
			if self.params['use_wandb'] and self.params['log_images_wandb']:
				image_array = sample.detach().cpu().numpy()
				image_array = np.transpose(image_array, (1, 2, 0))
				images = wandb.Image(image_array)
				wandb.log({"images": images})

		print('*************** Validation ***************')
		print('Expression Error: {:.4f}\t Pose Error: {:.2f}\t CSIM: {:.2f}'.format(exp_error/count, pose_error/count, csim_total/count))
		print('*************** Validation ***************')
		
		if self.params['use_wandb']:
			self.wandb.log({
				'expression_error': exp_error/count,
				'pose_error': pose_error/count,
				'csim': csim_total/count,
			})

	'Evaluate models for face reenactment and save reenactment figure for paired data training'
	def evaluate_model_reenactment_video(self, G, A_matrix, step):
		
		A_matrix.eval()
		input_is_latent = True
		exp_error = 0 ; pose_error = 0; csim_total = 0
		count = 0
		counter_logs = 0
		num_pairs_log = self.params['num_pairs_log']
		if self.reenactment_fig:
			source_images = torch.zeros((num_pairs_log, 3, 256, 256))
			target_images = torch.zeros((num_pairs_log, 3, 256, 256))
			reenacted_images = torch.zeros((num_pairs_log, 3, 256, 256))

		with torch.no_grad():
			for batch_idx, batch in enumerate(self.test_dataloader):
				if batch_idx > int(self.params['validation_samples'] / self.test_batch_size):
					print('Stop in ', batch_idx)
					break
				sample_batch = batch

				source_w = sample_batch['source_latent_code'].cuda()
				target_w = sample_batch['target_latent_code'].cuda()
				source_img = sample_batch['source_img'].cuda()
				target_img = sample_batch['target_img'].cuda()

				params_source, angles_source = calculate_shapemodel(self.deca, source_img)
				params_target, angles_target = calculate_shapemodel(self.deca, target_img)
				
				shift_vector = self.make_shifts_interpolation(params_source, params_target, angles_source, angles_target)
				shift = A_matrix(shift_vector)				
				imgs_shifted = generate_image(G, source_w, self.truncation, self.trunc, shift_code = shift, return_latents = False, input_is_latent = input_is_latent)
				params_shifted, angles_shifted = calculate_shapemodel(self.deca, imgs_shifted)

				csim, pose, exp = self.extract_evaluation_metrics(params_shifted, params_target, angles_shifted, angles_target, imgs_shifted, target_img)
				exp_error += exp
				csim_total += csim
				pose_error += pose
				count += 1

				if self.reenactment_fig:
					if counter_logs < num_pairs_log:
						if (num_pairs_log - counter_logs) % source_w.shape[0] == 0:
							source_images[counter_logs:counter_logs+source_w.shape[0]] = source_img.detach().cpu()
							target_images[counter_logs:counter_logs+source_w.shape[0]] = target_img.detach().cpu()
							reenacted_images[counter_logs:counter_logs+source_w.shape[0]] = imgs_shifted.detach().cpu()
						else:
							num = num_pairs_log - counter_logs
							source_images[counter_logs:counter_logs+num]  = source_img[:num].detach().cpu()
							target_images[counter_logs:counter_logs+num]  = target_img[:num].detach().cpu()
							reenacted_images[counter_logs:counter_logs+num]  = imgs_shifted[:num].detach().cpu()
						counter_logs += source_w.shape[0]

				

		if self.reenactment_fig:
			sample = generate_grid_image(source_images, target_images, reenacted_images)
			save_image(sample, os.path.join(self.images_reenact_dir, '{:06d}.png'.format(step)))
			if self.params['use_wandb'] and self.params['log_images_wandb']:
				image_array = sample.detach().cpu().numpy()
				image_array = np.transpose(image_array, (1, 2, 0))
				images = wandb.Image(image_array)
				wandb.log({"images": images})


		print('*************** Validation ***************')
		print('Expression Error: {:.4f}\t Pose Error: {:.2f}\t CSIM: {:.2f}'.format(exp_error/count, pose_error/count, csim_total/count))
		print('*************** Validation ***************')
		
		if self.params['use_wandb']:
			self.wandb.log({
				'expression_error': exp_error/count,
				'pose_error': pose_error/count,
				'csim': csim_total/count,
			})

		A_matrix.train()
	
	