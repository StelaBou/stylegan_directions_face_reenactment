import copy
import numpy as np
import torch
import glob
import sys
from tqdm import tqdm
sys.path.append(".")
sys.path.append("..")

from libs.criteria.PTI.localitly_regulizer import Space_Regulizer
from libs.criteria.PTI.base_coach import calc_loss
from libs.criteria.PTI import hyperparameters
from libs.criteria import id_loss
from libs.criteria import l2_loss
from libs.criteria.lpips.lpips import LPIPS



def print_losses(metrics_dict, step):
	out_text = '[step {}]'.format(step)
	for key, value in metrics_dict.items():
		out_text += (' | {}: {:.5f}'.format(key, value))
	print(out_text)

def optimize_g(generator, latent, real_imgs, opt_steps = 200, lr = 3e-3, use_ball_holder = False, optimize_all = False):
	trunc = generator.mean_latent(4096).detach().clone()
	truncation = 0.7
	generator_copy = copy.deepcopy(generator)
	generator.train()
	
	if not optimize_all: # Optimize some of the generator's parameters
		parameters = list(generator.convs[11].parameters()) + list(generator.convs[10].parameters()) + list(generator.convs[9].parameters()) \
			+ list(generator.convs[8].parameters()) + list(generator.convs[7].parameters()) + list(generator.convs[6].parameters()) \
			+ list(generator.convs[5].parameters()) + list(generator.convs[4].parameters()) 
		optimizer = torch.optim.Adam(parameters, lr=lr) 
		pt_l2_lambda = 100
	else:
		optimizer = torch.optim.Adam(generator.parameters(), lr=lr) 
		pt_l2_lambda = 1
	
	print('********** Start optimization for {} steps **********'.format(opt_steps))
	
	lpips_loss_fun = LPIPS(net_type='alex').cuda().eval()
	space_regulizer = Space_Regulizer(generator_copy, lpips_loss_fun)

	# for step in range(opt_steps):
	for step in tqdm(range(opt_steps)):

		loss_dict = {}
		imgs_gen, _ = generator([latent], input_is_latent=True, return_latents = False, truncation=truncation, truncation_latent=trunc)
		
		if use_ball_holder:
			loss, l2_loss_val, lpips_loss, ball_holder_loss_val = calc_loss(imgs_gen, real_imgs,
							generator, use_ball_holder, latent, space_regulizer, lpips_loss_fun, pt_l2_lambda)
			loss_dict["ball_holder_loss_val"] = ball_holder_loss_val.item()
		else:
			loss, l2_loss_val, lpips_loss = calc_loss(imgs_gen, real_imgs,
							generator, use_ball_holder, latent, space_regulizer, lpips_loss_fun, pt_l2_lambda)
		
		loss_dict['loss'] = loss.item()
		loss_dict["lpips_loss"] = lpips_loss.item()
		loss_dict['l2_loss'] = l2_loss_val.item() 
		
		# print_losses(loss_dict, step)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step() 
	
	print('********** End optimization **********')
		
	return generator	
