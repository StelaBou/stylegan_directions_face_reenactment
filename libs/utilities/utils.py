import os
import numpy as np
import torch
import json
from datetime import datetime
import glob

def get_image_files(path):
	types = ('*.png', '*.jpg') # the tuple of file types
	files_grabbed = []
	for files in types:
		files_grabbed.extend(glob.glob(os.path.join(path, files)))
	files_grabbed.sort()
	return files_grabbed

def get_files_frompath(path, types):
	files_grabbed = []
	for files in types:
		files_grabbed.extend(glob.glob(os.path.join(path, files)))
	files_grabbed.sort()
	return files_grabbed

def make_path(path):
	if not os.path.exists(path):
		os.makedirs(path, exist_ok = True)

def save_arguments_json(args, save_path, filename):
	out_json = os.path.join(save_path, filename)
	# datetime object containing current date and time
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	with open(out_json, 'w') as out:
		stat_dict = args
		json.dump(stat_dict, out)

def read_arguments_json(filename):
	with open(filename) as json_file:
		data = json.load(json_file)
		arguments_dict = data

	return arguments_dict

def delete_files(file_list):
    """Delete files with filenames in given list.
    Args:
        file_list (list): list of filenames to be deleted
    """
    for file in file_list:
        try:
            os.remove(file)
        except OSError:
            pass

def make_noise(batch, dim, truncation=None):
	if isinstance(dim, int):
		dim = [dim]
	if truncation is None or truncation == 1.0:
		return torch.randn([batch] + dim)
	else:
		return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)

def one_hot(dims, value, indx):
	vec = torch.zeros(dims)
	vec[indx] = value
	return vec
