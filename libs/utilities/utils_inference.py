import os
import numpy as np
import torch
from torchvision import utils as torch_utils
import cv2

from libs.utilities.image_utils import *
from libs.face_models.ffhq_cropping import crop_using_landmarks
from libs.utilities.utils import make_path

def generate_video(images, video_path, fps = 25):
	dim = (images[0].shape[1], images[0].shape[0])
	com_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V') , fps, dim)
	
	for image in images:
		com_video.write(np.uint8(image))
		
	com_video.release()

def generate_grid_image(source, target, reenacted):
	num_images = source.shape[0] # batch size
	width = 256; height = 256
	grid_image = torch.zeros((3, num_images*height, 3*width))
	for i in range(num_images):
		s = i*height
		e = s + height
		grid_image[:, s:e, :width] = source[i, :, :, :]
		grid_image[:, s:e, width:2*width] = target[i, :, :, :]	
		grid_image[:, s:e, 2*width:] = reenacted[i, :, :, :]
	
	if grid_image.shape[1] > 1000: # height
		grid_image = torch_image_resize(grid_image, height = 800)
	return grid_image

def extract_frames(video_path, fps = 25, save_frames = None, get_only_first = False):
	if not get_only_first:
		print('Extract frames with {} fps'.format(fps))
	if save_frames is not None:
		make_path(save_frames)
	cap = cv2.VideoCapture(video_path)
	counter = 0
	frames = []
	while cap.isOpened():
		ret, frame = cap.read()	
		if not ret:
			break	
		if get_only_first:
			return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)	
		if counter % fps == 0:
			if save_frames is not None:
				cv2.imwrite(os.path.join(save_frames, '{:06d}.png'.format(counter)), frame)
			frames.append( cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR))
		counter += 1

	cap.release()
	cv2.destroyAllWindows()

	return np.asarray(frames)

" Crop images using facial landmarks "
def preprocess_image(image_path, landmarks_est, save_filename = None):

	if os.path.isfile(image_path):
		image = read_image_opencv(image_path)
	else:
		image = image_path
	image, scale = image_resize(image, width = 1000)
	image_tensor = torch.tensor(np.transpose(image, (2,0,1))).float().cuda()	
	
	with torch.no_grad():
		landmarks = landmarks_est.detect_landmarks(image_tensor.unsqueeze(0))
		landmarks = landmarks[0].detach().cpu().numpy()
		landmarks = np.asarray(landmarks)

		img = crop_using_landmarks(image, landmarks)
		if img is not None and save_filename is not None:
			cv2.imwrite(save_filename,  cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))
		if img is not None:
			return img
		else:
			print('Error with image preprocessing')
			exit()

" Invert real image into the latent space of StyleGAN2 "
def invert_image(image, encoder, generator, truncation, trunc, save_path = None, save_name = None):
	with torch.no_grad():
		latent_codes = encoder(image)
		inverted_images, _ = generator([latent_codes], input_is_latent=True, return_latents = False, truncation= truncation, truncation_latent=trunc)

	if save_path is not None and save_name is not None:
		grid = torch_utils.save_image(
						inverted_images,
						os.path.join(save_path, '{}.png'.format(save_name)),
						normalize=True,
						range=(-1, 1),
					)
		# Latent code
		latent_code = latent_codes[0].detach().cpu().numpy()
		save_dir = os.path.join(save_path, '{}.npy'.format(save_name))
		np.save(save_dir, latent_code)

	return inverted_images, latent_codes