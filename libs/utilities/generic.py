import numpy as np
import torch
from numpy import ones,vstack
from numpy.linalg import lstsq
import os
import json
from torchvision import utils as torch_utils

from libs.configs.config_directions import get_direction_ranges, voxceleb_dict, ffhq_dict
from libs.utilities.image_utils import *


def save_image(image, save_image_path):
			
	grid = torch_utils.save_image(
		image,
		save_image_path,
		normalize=True,
		range=(-1, 1),
	)

def calculate_shapemodel(deca_model, images, image_space = 'gan'):
	img_tmp = images.clone()
	if image_space == 'gan':
		img_tmp = torch_range_1_to_255(img_tmp)
		
	p_tensor, alpha_shp_tensor, alpha_exp_tensor, angles, cam = deca_model.extract_DECA_params(img_tmp) # params dictionary 
	out_dict = {}
	out_dict['pose'] = p_tensor
	out_dict['alpha_exp'] = alpha_exp_tensor
	out_dict['alpha_shp'] = alpha_shp_tensor
	out_dict['cam'] = cam
	
	return out_dict, angles.cuda()

def initialize_directions(dataset_type, learned_directions, shift_scale):
	if dataset_type == 'voxceleb':
		ranges = get_direction_ranges(voxceleb_dict['ranges_filepath'])
		jaw_range = ranges[3]
		max_jaw = jaw_range[1]
		min_jaw = jaw_range[0]
		exp_ranges = ranges[4:]

		angle_scales = np.zeros(3)
		angle_scales[0] = voxceleb_dict['yaw_scale']
		angle_scales[1] = voxceleb_dict['pitch_scale']
		angle_scales[2] = voxceleb_dict['roll_scale']

		angle_directions = np.zeros(3)
		angle_directions[0] = int(voxceleb_dict['yaw_direction'])
		angle_directions[1] = int(voxceleb_dict['pitch_direction'])
		angle_directions[2] = int(voxceleb_dict['roll_direction'])

	else:
		angle_scales = np.zeros(3)
		angle_scales[0] = ffhq_dict['yaw_scale']
		angle_scales[1] = ffhq_dict['pitch_scale']
		angle_scales[2] = ffhq_dict['roll_scale']

		angle_directions = np.zeros(3)
		angle_directions[0] = ffhq_dict['yaw_direction']
		angle_directions[1] = ffhq_dict['pitch_direction']
		angle_directions[2] = ffhq_dict['roll_direction']
		exp_ranges = get_direction_ranges(ffhq_dict['ranges_filepath'])
		
		jaw_range = exp_ranges[3]
		jaw_range = jaw_range
		max_jaw = jaw_range[1]
		min_jaw = jaw_range[0]
		exp_ranges = exp_ranges[4:]

	directions_exp = []
	count_pose = 0
	if angle_directions[0] != -1:
		count_pose += 1
	if angle_directions[1] != -1:
		count_pose += 1
	if angle_directions[2] != -1:
		count_pose += 1
	count_pose += 1 # Jaw
	num_expressions = learned_directions - count_pose
	
	
	for i in range(num_expressions):
		dict_3d = {}
		dict_3d['exp_component'] = i
		dict_3d['A_direction'] = i + count_pose 
		dict_3d['max_shift'] =  exp_ranges[i][1]
		dict_3d['min_shift'] =  exp_ranges[i][0] 
		
		points = [(dict_3d['min_shift'], - shift_scale),(dict_3d['max_shift'], shift_scale)]
		x_coords, y_coords = zip(*points)
		A = vstack([x_coords,ones(len(x_coords))]).T
		m, c = lstsq(A, y_coords, rcond=None)[0]
		dict_3d['a'] = m
		dict_3d['b'] = c
		directions_exp.append(dict_3d)
	
		
	points = [(min_jaw, -6),(max_jaw, 6)]
	x_coords, y_coords = zip(*points)
	A = vstack([x_coords,ones(len(x_coords))]).T
	m, c = lstsq(A, y_coords, rcond=None)[0]
	a_jaw = m
	b_jaw = c

	jaw_dict = {
		'a':			a_jaw,
		'b':			b_jaw,
		'max':			max_jaw,
		'min':			min_jaw
	}
	
	return count_pose, num_expressions, directions_exp, jaw_dict, angle_scales, angle_directions

def get_shifted_latent_code(G, z, shift, input_is_latent = False, truncation=1, truncation_latent = None, w_plus = False, num_layers = None):		
	inject_index = G.n_latent
	if not input_is_latent: # Z space
		w = G.get_latent(z)
		latent = w.unsqueeze(1).repeat(1, inject_index, 1)
	else: # W space
		latent = z.clone()
	if not w_plus: # shift = B x 512		
		if num_layers is None: # add shift in all layers
			shift_rep = shift.unsqueeze(1)
			shift_rep = shift_rep.repeat(1, inject_index, 1)
			latent += shift_rep
		else:
			for i in range(num_layers):
				latent[:, i,:] += shift
	
	else:# shift= B x num_layers x 512
		latent[:, :shift.shape[1],:] += shift

	return latent

def generate_image( G, latent_code, truncation, trunc, w_plus = True, num_layers_shift = 8, shift_code = None, input_is_latent = False, return_latents = False):
	if shift_code is None:
		imgs = G([latent_code], return_latents = return_latents, truncation = truncation, truncation_latent = trunc, input_is_latent = input_is_latent)
	else:
		shifted_code = get_shifted_latent_code(G, latent_code, shift_code, input_is_latent = input_is_latent, truncation=truncation, 
			truncation_latent = trunc, w_plus = w_plus, num_layers = num_layers_shift)
		imgs = G([shifted_code], return_latents = return_latents, truncation = truncation, truncation_latent = trunc, input_is_latent = True)
	image = imgs[0]
	latent_w = imgs[1]
	face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
	if image.shape[2] > 256:
		image = face_pool(image)
	if return_latents:
		return image, latent_w
	else:
		return image