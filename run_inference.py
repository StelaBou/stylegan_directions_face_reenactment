import os
import datetime
import random
import sys
import warnings
import argparse 
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True

from libs.DECA.estimate_DECA import DECA_model
from libs.models.direction_matrix import DirectionMatrix
from libs.gan.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.face_models.landmarks_estimation import LandmarksEstimation
from libs.utilities.utils_inference import *
from libs.utilities.utils import make_path, get_image_files
from libs.gan.encoder4editing.psp_encoders import Encoder4Editing
from libs.optimization import optimize_g
from libs.utilities.image_utils import image_to_tensor, tensor_to_image
from libs.utilities.generic import initialize_directions, calculate_shapemodel, generate_image, save_image
from libs.configs.config_models import *

root_path =  os.getcwd()
seed = 0
random.seed(seed)

class Inference():

	def __init__(self, args):
		self.args = args
		self.device = 'cuda'

		self.source_path = args['source_path']
		self.target_path = args['target_path']
		self.output_path = args['output_path']
		make_path(self.output_path)

		self.dataset_type = args['dataset_type']
		self.image_resolution = args['image_resolution']
		self.encoder_path = args['encoder_path']
		self.A_matrix_model = args['A_matrix_model']

		self.save_grid = args['save_grid']
		self.save_images = args['save_images']
		self.optimize_generator = args['optimize_generator']
		self.save_video = args['save_video']
		
	def load_models(self):
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

		# Load inversion model
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

		head, tail = os.path.split(self.source_path)
		ext = tail.split('.')[-1]
		if ext != 'mp4' and ext != 'png' and ext != 'jpg':
			print('Wrong source path. Expected file image (.png, .jpg) or video (.mp4)')
			exit()

		self.input_is_latent = True
		if ext == 'mp4':
			self.source_path = extract_frames(self.source_path, get_only_first = True)

		# Step 1: Crop image
		cropped_image = preprocess_image(self.source_path, self.landmarks_est, save_filename = None)		
		# Step 2: Invert image
		source_image = image_to_tensor(cropped_image).unsqueeze(0).cuda()
		inv_image, source_code = invert_image(source_image, self.encoder, self.generator, self.truncation, self.trunc)
		# Step 3: Optimize generator
		if self.optimize_generator:
			G_copy = optimize_g(self.generator, source_code, source_image, optimize_all = False)
		else:
			G_copy = self.generator
		inverted_images, _ = G_copy([source_code], input_is_latent=True, return_latents = False, truncation= self.truncation, truncation_latent=self.trunc)
		# save_image(inverted_images, os.path.join(self.output_path, 'source_reconstructed.png'))
			
		return source_image, source_code, G_copy

	def load_target_data(self):
		image_files = None
		if os.path.isdir(self.target_path):
			image_files = get_image_files(self.target_path)
			if len(image_files) == 0:
				print('There are no images in {}'.format(self.target_path))
				exit()
			
		elif os.path.isfile(self.target_path):
			head, tail = os.path.split(self.target_path)
			ext = tail.split('.')[-1]
			# Check if file is image
			if ext == 'png' or ext == 'jpg':
				image_files = [self.target_path]
			# Check if file is image
			elif ext == 'mp4' or ext == 'avi':
				# Change FPS if needed
				image_files = extract_frames(self.target_path, fps = 1) # numpy arrays
			else:
				print('Please specify correct target path. Extension should be png, jpg, mp4 or avi')
				exit()
		else:
			print('Please specify correct target path: folder with images or image file or video')
			exit()

		return image_files

	def run_reenactment(self):
		self.load_models()
		source_img, source_code, G_source = self.load_source_data()
		target_images = self.load_target_data()
		print('Run reenactment for {} images'.format(len(target_images)))
		
		if self.save_grid:
			save_grid_path = os.path.join(self.output_path, 'grids')
			make_path(save_grid_path)

		grids = []
		### Calculate shape parameters for source image ###
		params_source, angles_source = calculate_shapemodel(self.shape_model, source_img)
		for i, target_image in enumerate(tqdm(target_images)):
			with torch.no_grad():
				### Load target image ###
				target_image = preprocess_image(target_image, self.landmarks_est, save_filename = None)
				target_image = image_to_tensor(target_image).unsqueeze(0).cuda()
				params_target, angles_target = calculate_shapemodel(self.shape_model, target_image)

				### Calculate shift vector ###
				shift_vector = self.make_shift(angles_source, angles_target, params_source, params_target).cuda()
				shift = self.A_matrix(shift_vector)	
				reenacted, shifted_latents = generate_image(G_source, source_code, self.truncation, self.trunc, self.w_plus, self.num_layers_shift, shift_code = shift, 
											input_is_latent= self.input_is_latent , return_latents=True)

			if self.save_images:
				save_dir = os.path.join(self.output_path, '{:06d}.png'.format(i))
				save_image(reenacted, save_dir)


			if self.save_grid or self.save_video:
				grid = generate_grid_image(source_img, target_image, reenacted)
				if self.save_grid:
					save_dir = os.path.join(save_grid_path, '{:06d}.png'.format(i))
					save_image(grid, save_dir)
				if self.save_video:
					image = cv2.cvtColor(tensor_to_image(grid), cv2.COLOR_BGR2RGB)
					grids.append(image)

		if self.save_video:
			save_dir = os.path.join(self.output_path, 'generated_video.mp4')
			generate_video(grids, save_dir)

	def make_shift(self, angles_source, angles_target, params_source, params_target):
		input_shift  = torch.zeros(self.learned_directions).cuda()
		# Source
		yaw_source = angles_source[:,0][0].detach().cpu().numpy() 
		pitch_source = angles_source[:,1][0].detach().cpu().numpy() 
		roll_source = angles_source[:, 2][0].detach().cpu().numpy() 
		exp_source = params_source['alpha_exp'][0].detach().cpu().numpy() 
		jaw_source = params_source['pose'][0, 3].detach().cpu().numpy()
		# Target
		yaw_target = angles_target[:,0][0].detach().cpu().numpy() 
		pitch_target = angles_target[:,1][0].detach().cpu().numpy() 
		roll_target = angles_target[:,2][0].detach().cpu().numpy() 
		exp_target = params_target['alpha_exp'][0].detach().cpu().numpy() 
		jaw_target = params_target['pose'][0, 3].detach().cpu().numpy()
		############################################################
		### Yaw ###
		ind = 0
		angles_source_shift_sc = yaw_source * self.shift_scale / self.angle_scales[0]
		angles_target_shift_sc = yaw_target * self.shift_scale / self.angle_scales[0]
		yaw_shift =  angles_target_shift_sc - angles_source_shift_sc
		input_shift[ind] = yaw_shift
		### Pitch ###	
		ind += 1		
		angles_source_shift_sc =  pitch_source * self.shift_scale / self.angle_scales[1]
		angles_target_shift_sc =  pitch_target * self.shift_scale / self.angle_scales[1]
		pitch_shift =  angles_target_shift_sc - angles_source_shift_sc
		input_shift[ind] = pitch_shift
		### Roll ###	
		ind += 1		
		angles_source_shift_sc =  roll_source * self.shift_scale / self.angle_scales[2]
		angles_target_shift_sc =  roll_target * self.shift_scale / self.angle_scales[2]
		roll_shift =  angles_target_shift_sc - angles_source_shift_sc
		input_shift[2] = roll_shift
		### Jaw ###
		ind += 1
		a = self.a_jaw
		b = self.b_jaw
		target = a * jaw_target + b
		source = a * jaw_source + b
		shift_exp = target - source
		input_shift[ind] = shift_exp
		### Expressions ###
		for index in range(self.num_expressions):		
			ind_exp = self.directions_exp[index]['exp_component']
			target_expression = exp_target[ind_exp]			
			source_expression = exp_source[ind_exp]		
			a = self.directions_exp[index]['a']
			b = self.directions_exp[index]['b']	
			target = a * target_expression + b
			source = a * source_expression + b
			shift_exp = target - source	
			input_shift[index + self.count_pose ] = shift_exp
		
		return input_shift.unsqueeze(0)			
	
	


def main():
	"""
	Inference script. Generate reenactment resuls.
	Input: 
		--source image or source video. If the input is video then get as source frame the first frame
		--target path with images, target image or target video
	Output:
		--save_images: save only the reenacted images 
		--save_grid:   save a grid with source, target, reenacted
		--save_video:  save a video with source, target, reenacted

	Options:
		######### General ###########
		--A_matrix_model		   			: model for inference

		--source_path						: Path to source identity. Type: image (.png or .jpg) or video (.mp4)
		--target_path						: Path to target poses. Type: image (.png or .jpg), video (.mp4) or path with images
		--output_path						: Path to save the results.

		--optimize_generator				: Optimize the generator. Default is True.

		######### Visualization #########
		--save_grid							: save source-target-reenacted grid
		--save_images						: save only reenacted images
		--save_video						: save results on video (source, target, reenacted)

	Example:

	python run_inference.py --source_path ./inference_examples/lWOTF8SdzJw#2614-2801.mp4 \
							--target_path ./inference_examples/lWOTF8SdzJw#2614-2801.mp4 \
							--output_path ./results --save_video

	
	
	"""
	parser = argparse.ArgumentParser(description="inference script")

	######### General ###########
	parser.add_argument('--A_matrix_model', type=str, default = './pretrained_models/A_matrix.pt', help="path to the A matrix model for inference")

	parser.add_argument('--source_path', type=str, required = True, help="path to source identity")
	parser.add_argument('--target_path', type=str, required = True, help="path to target pose")
	parser.add_argument('--output_path', type=str, required = True, help="path to save the results")

	######### Generator #########
	parser.add_argument('--image_resolution', type=int, default=256, choices=(256, 1024), help="image resolution of pre-trained GAN modeln")
	parser.add_argument('--dataset_type', type=str, default='voxceleb', choices=('voxceleb'), help="set dataset name")
	parser.add_argument('--encoder_path', type=str, default='./pretrained_models/e4e-voxceleb.pt', help="set pre-trained e4e model path")

	# Default is True. If False use argument
	parser.add_argument('--optimize_generator', dest='optimize_generator', action='store_false', help="Optimize generator") 
	parser.add_argument('--save_grid', dest='save_grid', action='store_true', help="save results on a grid (source, target, reenacted)")
	parser.set_defaults(save_grid=False)
	parser.add_argument('--save_images', dest='save_images', action='store_true', help="save reenacted images")
	parser.set_defaults(save_images=False)
	parser.add_argument('--save_video', dest='save_video', action='store_true', help="save results on video (source, target, reenacted)")
	parser.set_defaults(save_video=False)
	

	# Parse given arguments
	args = parser.parse_args()	
	args = vars(args) # convert to dictionary

	inference = Inference(args)
	inference.run_reenactment()
	


if __name__ == '__main__':
	main()

	
		
