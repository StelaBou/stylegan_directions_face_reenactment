import os
import numpy as np


voxceleb_dict = {

	'yaw_direction':		0,
	'pitch_direction':		1,
	'roll_direction':		2,
	'jaw_direction':		3,
	'yaw_scale':			40,
	'pitch_scale':			20,
	'roll_scale':			20,
	'ranges_filepath':		'./libs/configs/ranges_voxceleb.npy'
}


ffhq_dict = {

	'yaw_direction':		0,
	'pitch_direction':		1,
	'roll_direction':		-1,
	'jaw_direction':		3,
	'yaw_scale':			40,
	'pitch_scale':			20,
	'roll_scale':			20,
	'ranges_filepath':		'./libs/configs/ranges_FFHQ.npy'
}

def get_direction_ranges(range_filepath):
	
	if os.path.exists(range_filepath):
		exp_ranges = np.load(range_filepath) 
		exp_ranges = np.asarray(exp_ranges).astype('float64')
	else:
		print('{} does not exists'.format(range_filepath))
		exit()
	
	return exp_ranges

" For inference -> generate interpolation gifs"
def get_direction_info(direction_index, angle_directions, a_jaw, b_jaw, directions_exp, shift_scale, angle_scales, 
		count_pose, shifts_count, params_source, angles_source):
	if direction_index == angle_directions[0]:
		type_direction = 'yaw'; pose = True
		angle_scale = angle_scales[0]
		source_angle = angles_source[:,0][0].detach().cpu().numpy()
		max_angle = 30; min_angle = -30			
	elif direction_index == angle_directions[1]:
		type_direction = 'pitch'; pose = True
		angle_scale = angle_scales[1]
		source_angle = angles_source[:,1][0].detach().cpu().numpy()
		max_angle = 15; min_angle = -15		
	elif direction_index == angle_directions[2]:
		type_direction = 'roll'; pose = True
		angle_scale = angle_scales[2]
		source_angle = angles_source[:,2][0].detach().cpu().numpy()	
		max_angle = 15; min_angle = -15		
	else:
		if direction_index == count_pose - 1:
			type_direction = 'jaw'; pose = False
			jaw_exp_source = params_source['pose'][0, 3]
			start_pose = a_jaw * jaw_exp_source + b_jaw
			start_pose = start_pose.detach().cpu().numpy()
		else:				
			index = next((index for (index, d) in enumerate(directions_exp) if d['A_direction'] == direction_index), None)
			if index is not None:		
				ind_exp = directions_exp[index]['exp_component']
				type_direction = 'exp_{:02d}'.format(ind_exp); pose = False
				exp_target = params_source['alpha_exp'][0][ind_exp]
				a = directions_exp[index]['a']
				b = directions_exp[index]['b']
				start_pose = a * exp_target + b		
				start_pose = start_pose.detach().cpu().numpy()		
				
	if pose:
		start_pose = source_angle * shift_scale / angle_scale
		min_shift = (-shift_scale - start_pose)
		max_shift = (shift_scale - start_pose) + 1e-5
	else:
		min_shift = (-shift_scale - start_pose)
		max_shift = (shift_scale - start_pose) + 1e-5
	step = shift_scale / shifts_count

	return type_direction, start_pose, min_shift, max_shift, step