import os
import torch
import random
import sys
import warnings
import argparse 
import numpy as np
import imageio
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True

from libs.DECA.estimate_DECA import DECA_model
from libs.models.direction_matrix import DirectionMatrix
from libs.gan.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.face_models.landmarks_estimation import LandmarksEstimation
from libs.utilities.utils_inference import generate_grid_image, preprocess_image, invert_image
from libs.utilities.utils import make_path, one_hot, get_files_frompath, make_noise
from libs.gan.encoder4editing.psp_encoders import Encoder4Editing
from libs.optimization import optimize_g
from libs.utilities.image_utils import image_to_tensor, tensor_to_image, add_border
from libs.utilities.generic import initialize_directions, calculate_shapemodel, generate_image, save_image
from libs.configs.config_directions import get_direction_info
from libs.configs.config_models import *

seed = 0
random.seed(seed)
root_path =  os.getcwd()

class Inference_images():

	def __init__(self, args):
		self.args = args
		self.use_cuda = True
		self.device = 'cuda'

		self.source_path = args['source_path']
		self.directions = args['directions']
		self.output_path = args['output_path']
		make_path(self.output_path)
	
		self.image_resolution = args['image_resolution']
		self.dataset_type = args['dataset_type']
		self.encoder_path = args['encoder_path']
		self.A_matrix_model = args['A_matrix_model']

		self.optimize_generator = args['optimize_generator']
		self.save_gif = args['save_gif']
		self.save_images = args['save_images']
		
	def load_models(self, inversion = False):
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		self.shape_model = DECA_model('cuda')
		self.landmarks_est =  LandmarksEstimation(type = '2D')

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

		print('----- Load generator from {} -----'.format(self.gan_weights))
		self.truncation = 0.7		
		self.generator = StyleGAN2Generator(self.image_resolution, 512, 8, channel_multiplier= self.channel_multiplier)
		if self.image_resolution == 256:
			self.generator.load_state_dict(torch.load(self.gan_weights)['g_ema'], strict = False)
		else:
			self.generator.load_state_dict(torch.load(self.gan_weights)['g_ema'], strict = True)
		self.generator.cuda().eval()
		self.trunc = self.generator.mean_latent(4096).detach().clone()
		
		print('----- Load A matrix from {} -----'.format(self.A_matrix_model))
		state_dict = torch.load(self.A_matrix_model, map_location=torch.device('cpu'))
		self.learned_directions = state_dict['learned_directions']
		self.shift_scale = state_dict['shift_scale']
		self.w_plus = state_dict['w_plus']
		self.num_layers_shift = state_dict['num_layers_shift']
		self.A_matrix = DirectionMatrix(
				shift_dim= state_dict['shift_dim'],
				input_dim= self.learned_directions,
				out_dim= None,
				w_plus= self.w_plus, bias = True,  num_layers= self.num_layers_shift)
		self.A_matrix.load_state_dict(state_dict['A_matrix'])
		self.A_matrix.cuda().eval() 
		self.A_matrix.zero_grad()

		### Load inversion model only when the input is image. ###
		if inversion:
			print('----- Load e4e encoder from {} -----'.format(self.encoder_path))		
			self.encoder = Encoder4Editing(50, 'ir_se', self.image_resolution)
			ckpt = torch.load(self.encoder_path)
			self.encoder.load_state_dict(ckpt['e']) 
			self.encoder.cuda().eval()

		self.count_pose, self.num_expressions, self.directions_exp, jaw_dict, self.angle_scales, self.angle_directions = initialize_directions(self.dataset_type, 
				self.learned_directions, self.shift_scale)
		
		self.a_jaw = jaw_dict['a']
		self.b_jaw = jaw_dict['b']
		self.max_jaw = jaw_dict['max']
		self.min_jaw = jaw_dict['min']
	
	def load_source_data(self):
		inversion = False
		if self.source_path is None:
			# Generate random latent code
			files_grabbed = [make_noise(1, 512)]
		else:
			if os.path.isdir(self.source_path):
				files_grabbed = get_files_frompath(self.source_path, ['*.png', '*.jpg'] )
				if len(files_grabbed) == 0:
					files_grabbed = get_files_frompath(self.source_path, ['*.npy'])
					if len(files_grabbed) == 0:
						print('Please specify correct path: folder with images (.png, .jpg) or latent codes (.npy)')
						exit()
				else:
					inversion = True # invert real images

			elif os.path.isfile(self.source_path):

				head, tail = os.path.split(self.source_path)
				ext = tail.split('.')[-1]
				# Check if file is image
				if ext == 'png' or ext == 'jpg':
					files_grabbed = [self.source_path]
					inversion = True
				elif ext == 'npy':
					files_grabbed = [self.source_path]	
				else:
					print('Wrong source path. Expected file image (.png, .jpg) or latent code (.npy)')
					exit()
			else:
				print('Wrong source path. Expected file image (.png, .jpg) or latent code (.npy)')
				exit()

		return files_grabbed, inversion
	
	def get_shifted_image(self, G_copy, source_code, shift, direction_index):
		with torch.no_grad():
			shift_vector = one_hot(self.A_matrix.input_dim, shift, direction_index).cuda()
			latent_shift = self.A_matrix(shift_vector)	
			shifted_image = generate_image( G_copy, source_code, self.truncation, self.trunc, self.w_plus, shift_code=latent_shift, 
					input_is_latent = self.input_is_latent, num_layers_shift = self.num_layers_shift)
			if shifted_image.shape[2] > 256:
				shifted_image = self.face_pool(shifted_image)
		return shifted_image
	
	"""
	draw_red_box: draw a red box around the original image
	draw_original: draw original image on gif
	"""
	def interpolate(self, G_copy, source_img, source_code, direction_index, params_source, angles_source, i, draw_red_box = True, draw_original = True):	
		shifts_count = 10
		grid_ids = []; original_imgs = []

		type_direction, start_pose, min_shift, max_shift, step = get_direction_info(direction_index, self.angle_directions, self.a_jaw, self.b_jaw, self.directions_exp, 
								self.shift_scale, self.angle_scales, self.count_pose, shifts_count, params_source, angles_source)
		
		if self.save_images:
			output_path_images = os.path.join(self.output_path, 'images', '{}'.format(type_direction))
			make_path(output_path_images)

		print('Direction {}/{}: Start {:.3f} Min shift {:.3f} Max shift {:.3f} {}'.format(type_direction, direction_index, start_pose, min_shift, max_shift, step))		
		shifted_images = []; count = 0
		# min_shift --> start pose
		for shift in np.arange(min_shift, start_pose, step):
			shifted_image = self.get_shifted_image(G_copy, source_code, shift, direction_index)[0]
			shifted_images.append(shifted_image)
			if self.save_images and shift != start_pose:
				save_dir = os.path.join(output_path_images, '{}_{:03d}.png'.format(type_direction, count))
				save_image(shifted_image, save_dir)
			count += 1

		# Start pose -> max_shift
		for shift in np.arange(start_pose, max_shift, step):
			shifted_image = self.get_shifted_image(G_copy, source_code, shift, direction_index)[0]
			shifted_images.append(shifted_image)
			if self.save_images:
				if shift == start_pose and draw_red_box:
					shifted_image = add_border(shifted_image.clone())
				save_dir = os.path.join(output_path_images, '{}_{:03d}.png'.format(type_direction, count))
				save_image(shifted_image, save_dir)
			count += 1
		
		if self.save_gif:
			grids = []
			if draw_original:
				source_img_numpy = tensor_to_image(source_img[0].clone())
			for k in range(len(shifted_images)):
				im = shifted_images[k].squeeze(0)
				im = tensor_to_image(im)
				if draw_original:
					grid_im = np.zeros((256, 2*256, 3))
					grid_im[:, :256, :] = source_img_numpy	
					grid_im[:, 256:, :] = im	
					im = grid_im
				grids.append(im.astype(np.uint8))

			grids = np.array(grids) 
			fig_file_path = os.path.join(self.output_path, 'gif_{}_{:02d}.gif'.format(type_direction, i))	
			imageio.mimsave(fig_file_path, grids, fps = 15)

	def run_editing(self):
		self.directions = [int(i) for i in self.directions]
		self.directions = np.asarray(self.directions)
		if np.any(self.directions > 14):
			print('Please specify correct directions: Choices = [0-15]')
			exit()

		files, inversion = self.load_source_data()
		self.load_models(inversion)
		
		print('Run editing for {} images'.format(len(files)))
		shifted_images_per_id = []
		for i, filename in enumerate(tqdm(files)):
			if inversion: # Real image
				cropped_image = preprocess_image(filename, self.landmarks_est)		
				source_img = image_to_tensor(cropped_image).unsqueeze(0).cuda()
				inv_image, source_code = invert_image(source_img, self.encoder, self.generator, self.truncation, self.trunc)
				if self.optimize_generator:
					# Step 3: Optimize generator
					G_copy =  optimize_g(self.generator, source_code, source_img, optimize_all = False)
				else:
					G_copy = self.generator
				self.input_is_latent = True
			else: # synthetic latent code
				G_copy = self.generator
				if self.source_path is not None:
					source_code = torch.from_numpy(np.load(filename)).unsqueeze(0).cuda()
					self.input_is_latent = True # Assuming that the input latent code is on W or W+ space.
				else:
					source_code = filename.cuda()
					self.input_is_latent = False
				with torch.no_grad():
					source_img = generate_image( self.generator, source_code, self.truncation, self.trunc, self.w_plus, 
									input_is_latent = self.input_is_latent, num_layers_shift = self.num_layers_shift)
					if source_img.shape[2] > 256:
						source_img = self.face_pool(source_img)
			# Calculate shape model for source
			with torch.no_grad():
				params_source, angles_source = calculate_shapemodel(self.shape_model, source_img)
			
			for dim in self.directions:
				self.interpolate(G_copy, source_img, source_code, dim, params_source, angles_source, i)
				

if __name__ == '__main__':
	"""
	Facial image editing script. 
	Input: 
		--source image or source latent code in W or W+ space. If the input is None, then generate a random synthetic image.
		--list of directions to edit
			0:	yaw
			1:	pitch
			2:	roll
			3:	jaw (mouth open)
			4:	smile
			5-14: different expressions
	Output:
		

	Options:
		######### General ###########
		--A_matrix_model		   			: model for inference

		--source_path						: Path to source frame. Type: image (.png or .jpg) or latent (.npy) or None (generate random synthetic)
		--directions						: Which directions to edit [0-15]
		--output_path						: Path to save the results.

		--optimize_generator				: Optimize the generator. Default is False.

		######### Visualization #########
		--save_gif							: save gif with generated images
		--save_images						: save the generated images

	Example:

	python run_facial_editing.py --source_path ./inference_examples/0002775.png \
							--output_path ./results/facial_editing --save_gif --optimize_generator --directions 0 1 2 3 4

	"""
	parser = argparse.ArgumentParser(description="Facial image editing")
	######### General ###########
	parser.add_argument('--source_path', type=str, default = None, help="path to source image or latent code. If None generate a random synthetic image.")
	parser.add_argument('--directions', nargs='+', required=True, help='directions to edit')
	parser.add_argument('--output_path', type=str, required = True, help="path to save images")
	
	######### Generator #########
	parser.add_argument('--image_resolution', type=int, default=256, choices=(256, 1024), help="image resolution of pre-trained GAN modeln")
	parser.add_argument('--dataset_type', type=str, default='voxceleb', choices=('voxceleb', 'ffhq'), help="set dataset name")
	parser.add_argument('--encoder_path', type=str, default='./pretrained_models/e4e-voxceleb.pt', help="set pre-trained e4e model path")
	
	######### A matrix #########
	parser.add_argument('--A_matrix_model', type=str, default = './pretrained_models/A_matrix.pt', help="path to the A_matrix model")

	# Default is False. If True use argument
	parser.add_argument('--optimize_generator', dest='optimize_generator', action='store_true', help="Optimize generator") 
	parser.add_argument('--save_gif', dest='save_gif', action='store_true', help="save results on a gif")
	parser.set_defaults(save_gif=False)
	parser.add_argument('--save_images', dest='save_images', action='store_true', help="save images")
	parser.set_defaults(save_images=False)
	

	# Parse given arguments
	args = parser.parse_args()	
	args = vars(args) # convert to dictionary

	inference = Inference_images(args)
	inference.run_editing()

	
		
