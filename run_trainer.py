import os
import datetime
import random
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True

from libs.trainer import Trainer
from libs.configs.config_arguments import arguments

root_path =  os.getcwd()

def main():
	"""
	Training script.
	Options:
		######### General ###########
		--experiment_path		   			: path to save experiment
		--use_wandb		   		   			: use wandb to log losses and evaluation metrics
		--log_images_wandb	       			: if True log images on wandb	
		--project_wandb            			: Project name for wandb
		--resume_training_model				: Path to model to continue training or None

		######### Generator #########
		--dataset_type						: voxceleb or ffhq
		--image_resolution         			: image resolution of pre-trained GAN model. image resolution for voxceleb dataset is 256

		######### Dataset #########
		--synthetic_dataset_path        	: set synthetic dataset path for evaluation
		--train_dataset_path            	: set training dataset path
		--test_dataset_path         		: set testing dataset path
		
		######### Direction matrix A #########
		--training_method            		: set training method: 
												synthetic -> training only with synthetic images
												real -> training only with real images
												real_synthetic -> training with synthetic and real images
												paired -> training with paired images

		--lr								: set the learning rate of direction matrix model
		
		######### Training #########
		--max_iter	                		: set maximum number of training iterations
		--batch_size		   	   			: set training batch size
		
		Phase 1: Train with synthetic images only. Evaluation during training on synthetic images.
			python run_trainer.py --experiment_path ./training_attempts/exp_v00 --training_method synthetic 
		
		Phase 2: Train with both synthetic and real images. Evaluation during training on real images, source images are real target images are synthetic.
			python run_trainer.py --experiment_path ./training_attempts/exp_v00 --training_method real_synthetic \
			--train_dataset_path /datasets/VoxCeleb1/VoxCeleb_videos\
			--test_dataset_path /datasets/VoxCeleb1/VoxCeleb_videos_test

			python run_trainer.py --experiment_path ./training_attempts/test/exp_v00 --training_method paired --batch_size 4 \
			--train_dataset_path /home/stella/Desktop/datasets/VoxCeleb1/VoxCeleb_few_shot \
			--test_dataset_path /home/stella/Desktop/datasets/VoxCeleb1/VoxCeleb_few_shot --use_wandb --log_images_wandb

		Phase 3: Train with paired data. Evaluation during training on paired images.
			python run_trainer.py --experiment_path ./training_attempts/exp_v00 --training_method paired --batch_size 4 \
			--train_dataset_path /datasets/VoxCeleb1/VoxCeleb_videos \
			--test_dataset_path /datasets/VoxCeleb1/VoxCeleb_videos_test

	"""
	parser = argparse.ArgumentParser(description="training script")

	######### General ###########
	parser.add_argument('--experiment_path', type=str, required = True, help="path to save the experiment")
	parser.add_argument('--use_wandb', dest='use_wandb', action='store_true', help="use wandb to log losses and evaluation metrics")
	parser.set_defaults(use_wandb=False)
	parser.add_argument('--log_images_wandb', dest='log_images_wandb', action='store_true', help="if True log images on wandb")
	parser.set_defaults(log_images_wandb=False)
	parser.add_argument('--project_wandb', type=str, default = 'face-reenactment', help="Project name for wandb")

	parser.add_argument('--resume_training_model', type=str, default = None, help="Path to model or None")

	######### Generator #########
	parser.add_argument('--image_resolution', type=int, default=256, choices=(256, 1024), help="image resolution of pre-trained GAN modeln")
	parser.add_argument('--dataset_type', type=str, default='voxceleb', choices=('voxceleb', 'ffhq'), help="set dataset name")

	######### Dataset #########
	parser.add_argument('--synthetic_dataset_path', type=str, default=None, help="set synthetic dataset path for evaluation")
	parser.add_argument('--train_dataset_path', type=str, default=None, help="set training dataset path")
	parser.add_argument('--test_dataset_path', type=str, default=None, help="set testing dataset path")
	parser.add_argument('--training_method', type=str, default='synthetic',  choices=('synthetic', 'real', 'real_synthetic', 'paired'), help="set training method")
	parser.add_argument('--lr', type=float, default=0.0001, help=" set the learning rate of direction matrix model")
	
	######### Training #########
	parser.add_argument('--max_iter', type=int, default=100000, help="set maximum number of training iterations")
	parser.add_argument('--batch_size', type=int, default=12, help="set training batch size")
	parser.add_argument('--test_batch_size', type=int, default=2, help="set evaluation batch size")
	parser.add_argument('--workers', type=int, default=1, help="")
	
	
	# Parse given arguments
	args = parser.parse_args()	
	args = vars(args) # convert to dictionary

	args.update(arguments) # add arguments from libs.configs.config_arguments.py

	# Create output dir and save current arguments
	experiment_path = args['experiment_path']
	experiment_path = experiment_path + '_{}_{}'.format(args['dataset_type'], args['training_method'])
	args['experiment_path'] = experiment_path
	args['root_path'] = root_path
	# Set up trainer
	print("#. Experiment: {}".format(experiment_path))

	
	trainer = Trainer(args)

	training_method = args['training_method']
	if training_method == 'synthetic':	 
		trainer.train()
	elif training_method == 'real' or training_method == 'real_synthetic':
		trainer.train_real()
	elif training_method == 'paired':
		trainer.train_paired()
	

if __name__ == '__main__':
	main()




	