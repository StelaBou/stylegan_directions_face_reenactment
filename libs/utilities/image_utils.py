import torch
import numpy as np
# import scipy.misc
import cv2
import torchvision
import os

def torch_image_resize(image, width = None, height = None):
	dim = None
	(h, w) = image.shape[1:]
	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (height, int(w * r))
		scale = r
	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (int(h * r), width)
		scale = r
	image = image.unsqueeze(0)
	image = torch.nn.functional.interpolate(image, size=dim, mode='bilinear')
	return image.squeeze(0)
	

" Resize numpy array image"
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)
		scale = r

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))
		scale = r

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	return resized, scale
	
" Read image from path"
def read_image_opencv(image_path):
	img = cv2.imread(image_path, cv2.IMREAD_COLOR) # BGR order!!!!
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

	return img.astype('uint8')

"Transform image tensor to numpy array"
def im_to_numpy(img):
	img = img.detach().cpu().numpy()
	img = np.transpose(img, (1, 2, 0)) # H*W*C
	return img

"Transform numpy 255 to image torch div(255) 3 x H x W"
def numpy_255_to_torch_1(image):
	image_tensor = torch.tensor(np.transpose(image,(2,0,1))).float().div(255.0)	
	return image_tensor

" Trasnform torch tensor images from range [-1,1] to [0,255]"
def torch_range_1_to_255(image): 
	img_tmp = image.clone()
	min_val = -1
	max_val = 1
	img_tmp.clamp_(min=min_val, max=max_val)
	img_tmp.add_(-min_val).div_(max_val - min_val + 1e-5)
	img_tmp = img_tmp.mul(255.0)
	return img_tmp

" Trasnform torch tensor to numpy images from range [-1,1] to [0,255]"
def tensor_to_image(image_tensor):
	if image_tensor.ndim == 4:
		image_tensor = image_tensor.squeeze(0)

	min_val = -1
	max_val = 1
	image_tensor.clamp_(min=min_val, max=max_val)
	image_tensor.add_(-min_val).div_(max_val - min_val + 1e-5)
	image_tensor = image_tensor.mul(255.0).add(0.0) 

	image_tensor = image_tensor.detach().cpu().numpy()
	image_tensor = np.transpose(image_tensor, (1, 2, 0))

	return image_tensor

" Load image from file path to tensor [-1,1] range "
def image_to_tensor(image_file):
	max_val = 1
	min_val = -1
	if os.path.isfile(image_file):
		image = cv2.imread(image_file, cv2.IMREAD_COLOR) # BGR order
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('uint8')
	else:
		image = image_file
	if image.shape[0]>256:
		image, _ = image_resize(image, 256)
	image_tensor = torch.tensor(np.transpose(image,(2,0,1))).float().div(255.0)	
	image_tensor = image_tensor * (max_val - min_val) + min_val

	return image_tensor

" Draw a red rectangle around image "
def add_border(tensor):
	border = 3
	for ch in range(tensor.shape[0]):
		color = 1.0 if ch == 0 else -1
		tensor[ch, :border, :] = color
		tensor[ch, -border:,] = color
		tensor[ch, :, :border] = color
		tensor[ch, :, -border:] = color
	return tensor