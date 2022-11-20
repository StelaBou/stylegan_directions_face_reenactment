import numpy as np
import os

# Additional arguments for training

arguments = {
	'shift_scale':						6.0,				# set the maximum shift scale
	'min_shift':						0.1,				# set the minimum shift
	'learned_directions':				15,					# set the number of directions to learn
	'num_layers_shift':					8,					# set number of layers to add the shift
	'w_plus':							True,				# set w_plus True to find directions in the W+ space
	'disentanglement_50':				True,				# set True to train half images on the batch changing only one direction


	'lambda_identity':					10.0,				# identity loss weight
	'lambda_perceptual':				10.0,				# perceptual loss weight
	'lambda_pixel_wise':				1.0,				# pixel wise loss weight,  only on paired data
	'lambda_shape':						1.0,				# shape loss weight
	'lambda_mouth_shape':				1.0,				# mouth shape loss weight
	'lambda_eye_shape':					1.0,				# eye shape loss weight
	'lambda_w_reg':						0.0,				# w regularizer

	'steps_per_log':					10,					# set number iterations per log
	'steps_per_save':					1000,				# set number iterations per saving model
	'steps_per_ev_log':					1000,				# set number iterations per evaluation
	'validation_samples':				100, 				# number of samples for evaluation

	'reenactment_fig':					True,				# generate reenactment figure during evaluation
	'num_pairs_log':					4,					# how many pairs on the reenactment figure
	'gif':								False,				# generate gif with directions durion evaluation
	'evaluation':						True,				# evaluate model during training

}