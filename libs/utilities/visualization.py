import numpy as np
import torch
from PIL import Image
import io
import os
import cv2

from libs.utilities.utils import one_hot
from libs.utilities.generic import calculate_shapemodel, generate_image
from libs.configs.config_directions import get_direction_info
from libs.utilities.image_utils import tensor_to_image

def get_shifted_image(G, A_matrix, source_code, shift, direction_index, truncation, trunc, w_plus, input_is_latent, num_layers_shift):
	shift_vector = one_hot(A_matrix.input_dim, shift, direction_index).cuda()
	latent_shift = A_matrix(shift_vector)	
	shifted_image = generate_image( G, source_code, truncation, trunc, w_plus, shift_code=latent_shift, 
			input_is_latent = input_is_latent, num_layers_shift = num_layers_shift)
	return shifted_image

@torch.no_grad()
def make_interpolation_chart(G, A_matrix, shape_model, z, directions_exp, angle_scales, angle_directions, info, max_directions = None):

	input_is_latent = info['input_is_latent']
	learned_dims = info['learned_dims']
	count_pose = info['count_pose']
	a_jaw = info['a_jaw']
	b_jaw = info['b_jaw']
	shift_scale = info['shift_scale']
	shifts_count = info['shifts_count']
	w_plus = info['w_plus']
	num_layers_shift = info['num_layers_shift']

	grids = []; types = []
	truncation =  0.7
	trunc = G.mean_latent(4096).detach().clone()

	if max_directions is not None:
		if learned_dims > max_directions:
			learned_dims = max_directions

	# Generate original image and calculate shape model	
	original_img = generate_image( G, z, truncation, trunc, w_plus, input_is_latent = input_is_latent, 
		num_layers_shift = num_layers_shift, return_latents = False)	
	params_original, angles_original = calculate_shapemodel(shape_model, original_img)

	for direction_index in range(learned_dims):
		shifted_images = []
		type_direction, start_pose, min_shift, max_shift, step = get_direction_info(direction_index, angle_directions, a_jaw, b_jaw, directions_exp, 
								shift_scale, angle_scales, count_pose, shifts_count, params_original, angles_original)
		min_shift = (-shift_scale - start_pose)
		max_shift = (shift_scale - start_pose) + 1e-5
		# min_shift --> start pose
		for shift in np.arange(min_shift, start_pose, step):
			shifted_image = get_shifted_image(G, A_matrix, z, shift, direction_index, truncation, trunc, w_plus, input_is_latent, num_layers_shift)
			shifted_images.append(shifted_image[0].detach().cpu())

		# start pose -> max_shift
		for shift in np.arange(start_pose, max_shift, step):
			shifted_image = get_shifted_image(G, A_matrix, z, shift, direction_index, truncation, trunc, w_plus, input_is_latent, num_layers_shift)
			shifted_images.append(shifted_image[0].detach().cpu())

		grids.append(shifted_images)
		types.append(type_direction)

	
	if len(grids) > 0:
		for i in range(len(grids)):
			for j in range(len(grids[i])):
				im = grids[i][j]
				im = tensor_to_image(im)
				grids[i][j] = im.astype(np.uint8)

	return grids, types 