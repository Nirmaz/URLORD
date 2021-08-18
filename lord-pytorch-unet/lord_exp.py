import argparse
import dataset_unet
from assets import AssetManager
from model.training_unet import Lord
import os
import torch
from lord_unet import train_ulord
from experiments_unet import init_expe_directory, EXP_PATH, write_arguments_to_file, jason_dump, init_expe_directoryc, EXP_PATH_C, init_expe_directory_n, EXP_FP
import pickle
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os import listdir, mkdir
from os.path import join
CUDA_LAUNCH_BLOCKING=1
import wandb






def train_model(model_dict, path_exp, model_name, data_l_name, data_u_name, data_v_name, data_t_name, base_dir):


	train_ulord(args = None ,model_id = model_dict['model_id'], base_dir = base_dir, num_exp = model_dict['model_id'] +'_'+str(model_dict['dim']), path_exp = path_exp,
				  t_l = model_dict['train_model'], model_name = model_name, data_l_name= data_l_name,
				  data_u_name= data_u_name, data_v_name = data_v_name,
				  data_t_name = data_t_name, load_model = model_dict['load_model'], use_out_config = True,
				  path_config = model_dict['config_model'], path_unet_config = model_dict['config_unet'], dim = model_dict['dim'],
				  take_from_arg = False)




def evaluate_model(model_dict, path_exp, model_name, exp_dict, data_with_gt, data_param, split_dict):

	assets = AssetManager(exp_dict['base_dir'])
	model_dir = assets.get_model_dir(model_name)
	path_to_weights = join(model_dir, 'best_model.pth')
	path_results = join(path_exp,model_dict['model_id'] +'_'+str(model_dict['dim']))

	lord3d = Lord()

	lord3d.load(model_dir,model_dict['model_id'],model_dict['dim'], num_exp=None, path_exp=None)

	lord3d.evaluate(model_dict, exp_dict, path_to_weights, data_param, data_with_gt, path_results, split_dict)







def run_exp(args):

	num_exp = init_expe_directory_n(args.base_dir, init_dir = False)
	with open(os.path.join(args.exp_dict_dir, 'exp_dict.jason'), 'rb') as exp_dict_file:
		big_exp_dict = json.load(exp_dict_file)
	path_exp = join(args.base_dir, str(num_exp))
	mkdir(path_exp)
	path_save_config = join(path_exp,'config')
	mkdir(path_save_config)
	jason_dump(big_exp_dict, join(path_save_config ,'big_exp_dict.jason'))
	for exp_path in big_exp_dict['exps']:

		# load exp dict
		with open(os.path.join(exp_path, 'exp_dict.jason'), 'rb') as exp_dict_file:
			exp_dict = json.load(exp_dict_file)

		# print("exp_dict", exp_dict)
		path_new_exp = join(path_exp , exp_dict['exp_name'])
		mkdir(path_new_exp)
		path_save_config = join(path_new_exp, 'config')
		mkdir(path_save_config)
		jason_dump(exp_dict, join(path_save_config, exp_dict['exp_name'] + '_exp_dict.jason'))

		# load the config where the patches created
		with open(os.path.join(exp_dict['path_d_dict'], 'config.json'), 'rb') as patches_param_file:
			data_param = json.load(patches_param_file)

		# load the fetal_data_withgt
		with open(os.path.join(exp_dict['path_d'], 'fetal_data_withgt.h5'), 'rb') as patches_param_file:
			data_with_gt = pickle.load(patches_param_file)

		# load the split of cases
		with open(os.path.join(exp_dict['path_dataset'], 'param_dict.json'), 'rb') as patches_param_file:
			split_dict = json.load(patches_param_file)

		for model in exp_dict['models']:

			# load model dict parameter
			with open(os.path.join(exp_dict['models_path'], model), 'rb') as model_dict_file:
				model_dict = json.load(model_dict_file)

			#built a specific model for that certain dataset
			if model_dict['add_num_exp']:
				if model_dict['load_model']:
					raise Exception("you load model and change is name")
				model_name = model_dict['model_name'] + '_' + str(num_exp) + exp_dict['exp_name']
				model_dict['model_name'] = model_name
			else:
				model_name = model_dict['model_name']


			# create folder for model
			mkdir(join(path_new_exp, model_dict['model_id'] +'_' + str(model_dict['dim'])))
			mkdir(join(path_new_exp, model_dict['model_id'] +'_' + str(model_dict['dim']), 'results'))
			mkdir(join(path_new_exp, model_dict['model_id'] +'_' + str(model_dict['dim']), 'config'))
			mkdir(join(path_new_exp, model_dict['model_id'] +'_' + str(model_dict['dim']), 'logging'))
			jason_dump(model_dict, join(path_new_exp, model_dict['model_id'] +'_'+ str(model_dict['dim']), 'config', model_dict['model_id'] + '_exp_dict.jason'))


			train_model(model_dict, path_new_exp, model_name, exp_dict['data_l_name'], exp_dict['data_u_name'], exp_dict['data_v_name'], exp_dict['data_t_name'], exp_dict['base_dir'])

			if model_dict['evaluate_model']:
				evaluate_model(model_dict, path_new_exp, model_name, exp_dict, data_with_gt, data_param, split_dict)



def main():
	print(torch.cuda.is_available(), "cuda")
	# print("nir")
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)
	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True
	run_exp_parser = action_parsers.add_parser('run_exp')
	run_exp_parser.add_argument('-ed','--exp-dict-dir', type=str, required=True)
	run_exp_parser.set_defaults(func=run_exp)

	args = parser.parse_args()
	args.func(args)
	print("here")

if __name__ == '__main__':
	print("nir")
	# wandb.login(key=[your_api_key])
	# wandb.init(reinit=True)
	main()