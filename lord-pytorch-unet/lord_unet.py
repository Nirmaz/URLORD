
import argparse
import numpy as np
import dataset_unet
from assets import AssetManager
from model.training_unet import Lord
from config_unet import base_config, base_config_3d, base_config_unet
from model.config_umodel_unet import config_unet_nn, config_unet_nn_3d, config_unet_nn_3d_c
from model.training_comp_unet import UNet3D
import os
import torch
from experiments_unet import init_expe_directory, EXP_PATH, write_arguments_to_file, jason_dump, init_expe_directoryc, EXP_PATH_C
import sys
from os.path import join
import pickle
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CUDA_LAUNCH_BLOCKING=1
import wandb

def preprocess(args):

	assets = AssetManager(args.base_dir)
	img_dataset = dataset_unet.get_dataset(args.dataset_id, args.dataset_path)
	imgs, classes, segs = img_dataset.read_images(args.jumps)
	n_classes = np.unique(classes).size
	np.savez(
		file = assets.get_preprocess_file_path(args.data_name),
		imgs = imgs, classes = classes, segs = segs, n_classes = n_classes
	)

def preprocess_onegr(args):

	assets = AssetManager(args.base_dir)
	img_dataset = dataset_unet.get_dataset(args.dataset_id, args.dataset_path)

	Trufi = False
	Fiesta = False
	if args.t:
		Trufi = True

	if args.f:
		Fiesta = True

	imgs, classes, segs = img_dataset.read_images(load_trufi = Trufi, load_fiesta = Fiesta)
	n_classes = np.unique(classes).size
	np.savez(
		file = assets.get_preprocess_file_path(args.data_name),
		imgs = imgs, classes = classes, segs = segs, n_classes = n_classes
	)


def split_classes(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, classes, segs = data['imgs'], data['classes'], data['segs']

	n_classes = np.unique(classes).size
	test_classes = np.random.choice(n_classes, size=args.num_test_classes, replace=False)

	test_idx = np.isin(classes, test_classes)
	train_idx = ~np.isin(classes, test_classes)

	np.savez(
		file=assets.get_preprocess_file_path(args.test_data_name),
		imgs = imgs[test_idx], classes = classes[test_idx], segs = segs[test_idx], n_classes = n_classes
	)

	np.savez(
		file=assets.get_preprocess_file_path(args.train_data_name),
		imgs=imgs[train_idx], classes = classes[train_idx], segs= segs[train_idx], n_classes=n_classes
	)


def split_samples(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, classes, segs = data['imgs'], data['classes'], data['segs']

	n_classes = np.unique(classes).size
	n_samples = imgs.shape[0]

	n_test_samples = int(n_samples * args.test_split)

	test_idx = np.random.choice(n_samples, size=n_test_samples, replace=False)
	train_idx = ~np.isin(np.arange(n_samples), test_idx)

	np.savez(
		file=assets.get_preprocess_file_path(args.test_data_name),
		imgs = imgs[test_idx], classes = classes[test_idx], segs = segs[test_idx], n_classes = n_classes
	)

	np.savez(
		file=assets.get_preprocess_file_path(args.train_data_name),
		imgs=imgs[train_idx], classes = classes[train_idx], segs = segs[train_idx], n_classes=n_classes
	)


def split_samples_ulord(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, classes, segs = data['imgs'], data['classes'], data['segs']

	imgs_f, classes_f, segs_f = imgs[:int(imgs.shape[0] / 2)], classes[:int(imgs.shape[0] / 2)], segs[:int(imgs.shape[0] / 2)]
	imgs_t, classes_t, segs_t = imgs[int(imgs.shape[0] / 2):], classes[int(imgs.shape[0] / 2):], segs[int(imgs.shape[0] / 2):]

	print(np.where(classes_f == 1), "f == 1")
	print(np.where(classes_t == 0), "f == 0")

	n_classes = np.unique(classes).size
	n_samples = imgs.shape[0]

	n_labeld_f = int((n_samples / 2) * args.f_split)
	n_labeld_t = int((n_samples / 2) * args.t_split)

	labeld_idx_f = np.random.choice(n_labeld_f, size = n_labeld_f, replace=False)
	labeld_idx_t = np.random.choice(n_labeld_t, size = n_labeld_t, replace=False)

	unlabeld_idx_f = ~np.isin(np.arange((n_samples / 2)), labeld_idx_f)
	unlabeld_idx_t = ~np.isin(np.arange((n_samples / 2)), labeld_idx_t )

	labaled_img = np.concatenate((imgs_f[labeld_idx_f],imgs_t[labeld_idx_t]))
	labaled_class = np.concatenate((classes_f[labeld_idx_f],classes_t[labeld_idx_t]))
	labaled_seg = np.concatenate((segs_f[labeld_idx_f],segs_t[labeld_idx_t]))
	print(imgs_f[labeld_idx_f].shape, "imgs_f[labeld_idx_f]")
	print(imgs_t[labeld_idx_t].shape, "imgs_t[labeld_idx_t]")

	img_u = np.concatenate((imgs_f[unlabeld_idx_f], imgs_t[unlabeld_idx_t]))
	class_u = np.concatenate((classes_f[unlabeld_idx_f], classes_t[unlabeld_idx_t]))
	seg_u = np.concatenate((segs_f[unlabeld_idx_f], segs_t[unlabeld_idx_t]))
	print(img_u.shape , "img_u")
	print(labaled_img.shape , "labaled_img")
	np.savez(
		file=assets.get_preprocess_file_path(args.labled_data_name),
		imgs = labaled_img , classes = labaled_class , segs = labaled_seg, n_classes = n_classes
	)

	np.savez(
		file=assets.get_preprocess_file_path(args.ulabled_data_name),
		imgs = img_u, classes = class_u, segs = seg_u, n_classes=n_classes
	)


def train_unetlatent_model(args):

	num_exp = init_expe_directory()
	write_arguments_to_file(sys.argv[4:], join(EXP_PATH, str(num_exp), 'config'))

	assets = AssetManager(args.base_dir)
	data = np.load(assets.get_preprocess_file_path(args.data_name))
	data_v = np.load(assets.get_preprocess_file_path(args.data_uname))
	# print(data.keys(), "data keys")
	imgs = data['imgs'].astype(np.float32)
	imgs_val = data_v['imgs'].astype(np.float32)
	if args.load_model:
		model_dir = assets.get_model_dir(args.model_name)
		tensorboard_dir = assets.get_tensorboard_dir(args.model_name)
		lord = Lord()
		lord.load(model_dir,unetlatent = True, num_exp = str(num_exp))
	else:
		model_dir = assets.recreate_model_dir(args.model_name)
		tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

		n_imgs = imgs.shape[0]
		n_classes = data['n_classes'].item()

		config = dict(
			num_exp=str(num_exp),
			img_shape=imgs.shape[1:],
			n_imgs=n_imgs,
			n_classes=n_classes,
			seg_loss= args.seg_loss,
			recon_loss=args.recon_loss,
			Regularized_content=args.r_content,
			Regularized_class=args.r_rclass,
		)

		config.update(base_config)
		path_config = join(EXP_PATH, str(num_exp), 'config', 'config.jason')
		path_config_unet = join(EXP_PATH, str(num_exp), 'config', 'config_unet.jason')
		jason_dump(config, path_config)
		jason_dump(config_unet_nn, path_config_unet)

		lord = Lord(config, config_unet_nn)

	lord.train_UnetLatentModel(
		imgs=imgs,
		segs = data['segs'],
		classes=data['classes'],
		imgs_val=imgs_val,
		segs_val =data_v['segs'],
		classes_val =data_v['classes'],
		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir,
		loaded_model= args.load_model
	)

	lord.save(model_dir, unetlatent = True)


def train_ulord(args):

	num_exp = init_expe_directory()
	write_arguments_to_file(sys.argv[4:], join(EXP_PATH, str(num_exp), 'config'))

	assets = AssetManager(args.base_dir)
	data = np.load(assets.get_preprocess_file_path(args.data_l_name))
	data_u = np.load(assets.get_preprocess_file_path(args.data_u_name))
	imgs = data['imgs'].astype(np.float32)
	imgs_u = data_u['imgs'].astype(np.float32)

	if args.g_config:
		with open(os.path.join(args.p_config, 'config.pkl'), 'rb') as config_fd:
			base_config = pickle.load(config_fd)

		with open(os.path.join(args.p_uconfig, 'config_unet.pkl'), 'rb') as config_fd:
			config_unet_nn = pickle.load(config_fd)

	if args.g_config and args.load_model == 0:
		with open(os.path.join(args.p_config, 'config.pkl'), 'rb') as config_fd:
			c_base_config = pickle.load(config_fd)

		with open(os.path.join(args.p_uconfig, 'config_unet.pkl'), 'rb') as config_fd:
			c_config_unet_nn = pickle.load(config_fd)
	else:
		c_base_config_3d = base_config_3d
		c_config_unet_nn_3d = config_unet_nn_3d

	if args.load_model:
		model_dir = assets.get_model_dir(args.model_name)
		tensorboard_dir = assets.get_tensorboard_dir(args.model_name)
		lord = Lord()
		lord.load(model_dir,ulord= True,ulord3d = False, num_exp = str(num_exp))
	else:
		model_dir = assets.recreate_model_dir(args.model_name)
		tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

		n_imgs = imgs.shape[0]
		n_classes = data['n_classes'].item()

		config = dict(
			num_exp=str(num_exp),
			img_shape=imgs.shape[1:],
			n_imgs=n_imgs,
			n_classes=n_classes,
			seg_loss= args.seg_loss,
			segmentor_gard = args.seg_gard,
			recon_loss=args.recon_loss,
			Regularized_content=args.r_content,
			Regularized_class=args.r_rclass,
		)

		config.update(c_base_config)
		path_config = join(EXP_PATH, str(num_exp), 'config', 'config.jason')
		path_config_unet = join(EXP_PATH, str(num_exp), 'config', 'config_unet.jason')
		jason_dump(config, path_config)
		jason_dump(c_config_unet_nn, path_config_unet)

		lord = Lord(config, c_config_unet_nn)

	lord.train_ULordModel(
		imgs=imgs,
		segs = data['segs'],
		classes=data['classes'],
		imgs_u=imgs_u,
		segs_u = data_u['segs'],
		classes_u = data_u['classes'],
		model_dir = model_dir,
		tensorboard_dir=tensorboard_dir,
		loaded_model= args.load_model
	)

	lord.save(model_dir, ulord= True,ulord3d = False )

def train_ulord3d(args):

	num_exp = init_expe_directory()
	write_arguments_to_file(sys.argv[4:], join(EXP_PATH, str(num_exp), 'config'))

	assets = AssetManager(args.base_dir)
	data = np.load(assets.get_preprocess_file_path(args.data_l_name))
	data_u = np.load(assets.get_preprocess_file_path(args.data_u_name))
	data_v = np.load(assets.get_preprocess_file_path(args.data_v_name))
	imgs = data['imgs'].astype(np.float32)
	imgs_u = data_u['imgs'].astype(np.float32)
	imgs_v = data_v['imgs'].astype(np.float32)

	if args.g_config and args.load_model == 0:

		with open(os.path.join(args.p_config, 'config.jason'), 'rb') as config_fd:
			c_base_config_3d = json.load(config_fd)

		with open(os.path.join(args.p_uconfig, 'config_unet.jason'), 'rb') as config_fd:
			c_config_unet_nn_3d = json.load(config_fd)
	else:
		c_base_config_3d = base_config_3d
		c_config_unet_nn_3d = config_unet_nn_3d




	if args.load_model:
		model_dir = assets.get_model_dir(args.model_name)
		tensorboard_dir = assets.get_tensorboard_dir(args.model_name)
		lord = Lord()
		lord.load(model_dir,ulord= False ,ulord3d = True, num_exp = str(num_exp))
	else:
		model_dir = assets.recreate_model_dir(args.model_name)
		tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

		n_imgs = imgs.shape[0]
		n_classes = data['n_classes'].item()

		config = dict(
			num_exp=str(num_exp),
			img_shape=imgs.shape[1:],
			n_imgs=n_imgs,
			n_classes=n_classes,
			seg_loss= args.seg_loss,
			segmentor_gard = args.seg_gard,
			recon_loss=args.recon_loss,
			Regularized_content=args.r_content,
			Regularized_class=args.r_rclass,
		)


		config.update(c_base_config_3d)
		path_config = join(EXP_PATH, str(num_exp), 'config', 'config.jason')
		path_config_unet = join(EXP_PATH, str(num_exp), 'config', 'config_unet.jason')
		jason_dump(config, path_config)
		jason_dump(c_config_unet_nn_3d, path_config_unet)
		lord = Lord(config, c_config_unet_nn_3d)

	lord.train_ULordModel3D(
		imgs=imgs,
		segs = data['segs'],
		classes=data['classes'],
		imgs_u=imgs_u,
		segs_u = data_u['segs'],
		classes_u = data_u['classes'],
		imgs_v = imgs_v,
		segs_v = data_v['segs'],
		classes_v = data_v['classes'],
		model_dir = model_dir,
		tensorboard_dir=tensorboard_dir,
		loaded_model= args.load_model
	)

	lord.save(model_dir, ulord = False, ulord3d = True)


def train_unet3D(args):

	num_exp = init_expe_directoryc()
	write_arguments_to_file(sys.argv[4:], join(EXP_PATH_C, str(num_exp), 'config'))

	assets = AssetManager(args.base_dir)
	data = np.load(assets.get_preprocess_file_path(args.data_l_name))
	data_v = np.load(assets.get_preprocess_file_path(args.data_v_name))
	imgs = data['imgs'].astype(np.float32)
	imgs_v = data_v['imgs'].astype(np.float32)

	if args.g_config and args.load_model == 0:
		with open(os.path.join(args.p_config, 'config.jason'), 'rb') as config_fd:
			c_base_config_3d = json.load(config_fd)

		with open(os.path.join(args.p_uconfig, 'config_unet.jason'), 'rb') as config_fd:
			c_config_unet_nn_3d = json.load(config_fd)
	else:
		c_base_config_3d = base_config_unet
		c_config_unet_nn_3d = config_unet_nn_3d_c

	if args.load_model:
		model_dir = assets.get_model_dir(args.model_name)
		tensorboard_dir = assets.get_tensorboard_dir(args.model_name)
		unet = UNet3D()
		unet.load(model_dir,unet = True, num_exp = str(num_exp))
	else:
		model_dir = assets.recreate_model_dir(args.model_name)
		tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

		n_imgs = imgs.shape[0]
		n_classes = data['n_classes'].item()

		config = dict(
			num_exp=str(num_exp),
		)

		config.update(c_base_config_3d)
		path_config = join(EXP_PATH_C, str(num_exp), 'config', 'config.jason')
		path_config_unet = join(EXP_PATH_C, str(num_exp), 'config', 'config_unet.jason')
		jason_dump(config, path_config)
		jason_dump(c_config_unet_nn_3d, path_config_unet)

		unet = UNet3D(config, c_config_unet_nn_3d)

	unet.train_UNet3D(
		imgs=imgs,
		segs = data['segs'],
		classes=data['classes'],
		imgs_v = imgs_v,
		segs_v = data_v['segs'],
		classes_v = data_v['classes'],
		model_dir = model_dir,
		tensorboard_dir=tensorboard_dir,
		loaded_model= args.load_model
	)

	unet.save(model_dir, unet = True)


def train_encoders(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	tensorboard_dir = assets.get_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['imgs'].astype(np.float32) / 255.0

	lord = Lord()
	lord.load(model_dir, latent=True, amortized=False)

	lord.train_amortized(
		imgs=imgs,
		classes=data['classes'],
		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir
	)

	lord.save(model_dir, latent=False, amortized=True)

def main():
	print(torch.cuda.is_available(), "cuda")

	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset_unet.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=False)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.add_argument('-j', '--jumps', type=int, required=True)
	preprocess_parser.set_defaults(func = preprocess)

	preprocess_parser = action_parsers.add_parser('preprocess_onegr')
	preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset_unet.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=False)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.add_argument('-t', '--t', type=int, required=True)
	preprocess_parser.add_argument('-f', '--f', type=int, required=True)
	# preprocess_parser.add_argument('-j', '--jumps', type=int, required=True)
	preprocess_parser.set_defaults(func = preprocess_onegr)

	split_classes_parser = action_parsers.add_parser('split-classes')
	split_classes_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	split_classes_parser.add_argument('-trdn', '--train-data-name', type=str, required=True)
	split_classes_parser.add_argument('-tsdn', '--test-data-name', type=str, required=True)
	split_classes_parser.add_argument('-ntsi', '--num-test-classes', type=int, required=True)
	split_classes_parser.set_defaults(func = split_classes)

	split_samples_parser = action_parsers.add_parser('split-samples')
	split_samples_parser.add_argument('-idn', '--input-data-name', type = str, required=True)
	split_samples_parser.add_argument('-trdn', '--train-data-name', type = str, required=True)
	split_samples_parser.add_argument('-tsdn', '--test-data-name', type = str, required=True)
	split_samples_parser.add_argument('-ts', '--test-split', type = float, required=True)
	split_samples_parser.set_defaults(func = split_samples)

	split_samples_parser = action_parsers.add_parser('split-samples-ulord')
	split_samples_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	split_samples_parser.add_argument('-lrdn', '--labled-data-name', type=str, required=True)
	split_samples_parser.add_argument('-tsdn', '--ulabled-data-name', type=str, required=True)
	split_samples_parser.add_argument('-fs', '--f-split', type=float, required=True)
	split_samples_parser.add_argument('-ts', '--t-split', type=float, required=True)
	split_samples_parser.set_defaults(func = split_samples_ulord)


	# train_parser = action_parsers.add_parser('train')
	# train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	# train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	# train_parser.add_argument('-lm', '--load-model', type=int, required=True)
	# # train_parser.add_argument('-cc', '--class-constant', type=int, required=True)
	# # train_parser.add_argument('-sc', '--source-class', type=str, required=False)
	# train_parser.add_argument('-rc', '--r-class', type=int, required=False)
	# train_parser.add_argument('-rr', '--r-rcontent', type=int, required=False)
	# train_parser.add_argument('-ot', '--opp-t', type=int, required=False)
	# train_parser.set_defaults(func = train)

	train_semi_parser = action_parsers.add_parser('train_ulord')
	train_semi_parser.add_argument('-dn', '--data-l-name', type=str, required=True)
	train_semi_parser.add_argument('-dun', '--data-u-name', type=str, required=True)
	train_semi_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_semi_parser.add_argument('-lm', '--load-model', type=int, required=True)
	train_semi_parser.add_argument('-sl', '--seg-loss', type=int, required=True)
	train_semi_parser.add_argument('-sg', '--seg-gard', type=int, required=True)
	train_semi_parser.add_argument('-rl', '--recon-loss', type=int, required=True)
	train_semi_parser.add_argument('-rc', '--r-content', type=int, required=True)
	train_semi_parser.add_argument('-rr', '--r-rclass', type=int, required=True)
	train_semi_parser.add_argument('-gc', '--g-config', type=int, required=True)
	train_semi_parser.add_argument('-pc', '--p-config', type=str, required=True)
	train_semi_parser.add_argument('-pu', '--p-uconfig', type=str, required=True)
	# train_parser.add_argument('-cc', '--class-constant', type=int, required=True)
	# train_parser.add_argument('-sc', '--source-class', type=str, required=False)
	train_semi_parser.set_defaults(func = train_ulord)

	train_semi_parser = action_parsers.add_parser('train_ulord3d')
	train_semi_parser.add_argument('-dn', '--data-l-name', type=str, required=True)
	train_semi_parser.add_argument('-dun', '--data-u-name', type=str, required=True)
	train_semi_parser.add_argument('-dv', '--data-v-name', type=str, required=True)
	train_semi_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_semi_parser.add_argument('-lm', '--load-model', type=int, required=True)
	train_semi_parser.add_argument('-sl', '--seg-loss', type=int, required=True)
	train_semi_parser.add_argument('-sg', '--seg-gard', type=int, required=True)
	train_semi_parser.add_argument('-rl', '--recon-loss', type=int, required=True)
	train_semi_parser.add_argument('-rc', '--r-content', type=int, required=True)
	train_semi_parser.add_argument('-rr', '--r-rclass', type=int, required=True)
	train_semi_parser.add_argument('-gc', '--g-config', type=int, required=True)
	train_semi_parser.add_argument('-pc', '--p-config', type=str, required=True)
	train_semi_parser.add_argument('-pu', '--p-uconfig', type=str, required=True)
	# train_parser.add_argument('-cc', '--class-constant', type=int, required=True)
	# train_parser.add_argument('-sc', '--source-class', type=str, required=False)
	train_semi_parser.set_defaults(func = train_ulord3d)


	train_semi_parser = action_parsers.add_parser('train_unet3D')
	train_semi_parser.add_argument('-dn', '--data-l-name', type=str, required=True)
	train_semi_parser.add_argument('-dv', '--data-v-name', type=str, required=True)
	train_semi_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_semi_parser.add_argument('-lm', '--load-model', type=int, required=True)
	train_semi_parser.add_argument('-gc', '--g-config', type=int, required=True)
	train_semi_parser.add_argument('-pc', '--p-config', type=str, required=True)
	train_semi_parser.add_argument('-pu', '--p-uconfig', type=str, required=True)
	# train_parser.add_argument('-cc', '--class-constant', type=int, required=True)
	# train_parser.add_argument('-sc', '--source-class', type=str, required=False)
	train_semi_parser.set_defaults(func = train_unet3D)



	train_semi_parser = action_parsers.add_parser('train_ulm')
	train_semi_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_semi_parser.add_argument('-dv', '--data-uname', type=str, required=True)
	train_semi_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_semi_parser.add_argument('-lm', '--load-model', type=int, required=True)
	train_semi_parser.add_argument('-sl', '--seg-loss', type=int, required=True)
	train_semi_parser.add_argument('-sg', '--seg-gard', type=int, required=True)
	train_semi_parser.add_argument('-rl', '--recon-loss', type=int, required=True)
	train_semi_parser.add_argument('-rc', '--r-content', type=int, required=True)
	train_semi_parser.add_argument('-rr', '--r-rclass', type=int, required=True)
	# train_parser.add_argument('-cc', '--class-constant', type=int, required=True)
	# train_parser.add_argument('-sc', '--source-class', type=str, required=False)
	train_semi_parser.set_defaults(func = train_unetlatent_model)



	train_encoders_parser = action_parsers.add_parser('train-encoders')
	train_encoders_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_encoders_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_encoders_parser.set_defaults(func = train_encoders)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	# wandb.login(key=[your_api_key])
	wandb.init(reinit=True)
	main()
