import abc
import os
import pickle
from argparse import Namespace
import wandb
import os.path
import sys

sys.path.append( '.' )
sys.path.append( '..' )

import torch
from torchvision import transforms
from libs.criteria.PTI.localitly_regulizer import Space_Regulizer
from libs.criteria.PTI import hyperparameters

l2_criterion = torch.nn.MSELoss(reduction='mean')


def l2_loss(real_images, generated_images):
    loss = l2_criterion(real_images, generated_images)
    return loss

def calc_loss(generated_images, real_images, new_G, use_ball_holder, w_batch, space_regulizer, lpips_loss, pt_l2_lambda):
    loss = 0.0

    # if hyperparameters.pt_l2_lambda > 0:
    if pt_l2_lambda > 0:
        l2_loss_val = l2_loss(generated_images, real_images)
        loss += l2_loss_val * pt_l2_lambda
    if hyperparameters.pt_lpips_lambda > 0:
        loss_lpips = lpips_loss(generated_images, real_images)
        loss_lpips = torch.squeeze(loss_lpips)

        loss += loss_lpips * hyperparameters.pt_lpips_lambda

    if use_ball_holder:
        ball_holder_loss_val = space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=False)
        loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips, ball_holder_loss_val
    else:
        return loss, l2_loss_val, loss_lpips
