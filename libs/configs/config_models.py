import os
import numpy as np

stylegan2_voxceleb_256 = {
	'image_resolution':			256,
	'channel_multiplier':		1,
	'gan_weights':				'./pretrained_models/stylegan-voxceleb.pt',
}

stylegan2_ffhq_256 = {
	'image_resolution':			256,
	'channel_multiplier':		2,
	'gan_weights':				'/home/stella/Desktop/projects/Finding_Directions_Reenactment/pretrained_models/stylegan_rosinality.pt',
}

stylegan2_ffhq_1024 = {
	'image_resolution':			1024,
	'channel_multiplier':		2,
	'gan_weights':				'/home/stella/Desktop/projects/Finding_Directions_Reenactment/pretrained_models/stylegan2-ffhq-config-f_1024.pt',
}

