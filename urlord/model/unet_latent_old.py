
import itertools
import pickle
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from model.modules_unet import VGGDistance, ULordModel, UnetLatentModel, Segmentntor, ULordSModel
from model.utils_unet import AverageMeter, NamedTensorDataset, ImbalancedDatasetSampler, MyRotationTransform
from PIL import Image
import os
import matplotlib.pyplot as plt
from os.path import join
from experiments_unet import EXP_PATH, jason_dump
from scipy.spatial import distance
CUDA_LAUNCH_BLOCKING=1
import wandb
import logging

def define_logger(loger_name, file_path):
	logger = logging.getLogger(loger_name)
	logger.setLevel(logging.INFO)
	file_handler = logging.FileHandler(file_path)
	logger.addHandler(file_handler)
	return logger




def dice_loss(inputs, targets, smooth = 1):
	"""This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

	# inputs = F.sigmoid(inputs)
	# flatten label and prediction tensors
	inputs = inputs.view(-1)
	targets = targets.view(-1)
	intersection = (inputs * targets).sum()
	dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
	return 1 - dice


def parts(path):
	"""
	parser to the path
	:param path:
	:return:
	"""
	components = []
	while True:
		(path,tail) = os.path.split(path)
		if tail == "":
			components.reverse()
			return components
		components.append(tail)

def pickle_load(in_file):
	with open(in_file, "rb") as opened_file:
		return pickle.load(opened_file)

class Lord:

	def __init__(self, config=None, config_unet = None):
		super().__init__()

		self.config_unet = config_unet
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.unet_latent_model = None
		self.ulord_model = None
		# self.segmentor = Segmentntor(config_unet)
		# self.amortized_model = None
		# self.semi_amortized_model = None
		self.seq_classifier = torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Dropout(p=0.2),
			torch.nn.Linear(128 * 128, 512, bias=True),
			torch.nn.ReLU(),
			torch.nn.Dropout(p=0.5),
			torch.nn.Linear(512, 256, bias=True),
			torch.nn.ReLU(),
			torch.nn.Dropout(p=0.5),
			torch.nn.Linear(256, 128, bias=True),
			torch.nn.ReLU(),
			torch.nn.Dropout(p=0.5),
			torch.nn.Linear(128, 2, bias=True),
			torch.nn.ReLU(),
			torch.nn.Dropout(p=0.5),
			torch.nn.Linear(2, 1, bias=True),
			torch.nn.Sigmoid())
		self.seq_classifier.load_state_dict(
			torch.load('/cs/casmip/nirm/embryo_project_version1/classfier_weights/model_weights2b.pth'))

		# if not config['class_code_constant']:
		# 	dict_class_embedding = pickle_load(config['source'])
		# 	class_arr = np.zeros((config['n_classes'], config['class_dim']))
		# 	for key in dict_class_embedding.keys():
		# 		class_arr[int(key), :] = dict_class_embedding[key]
		#
		# 	self.class_embedding_tesnor = torch.from_numpy(class_arr)

	def load(self, model_dir, unetlatent = True,ulord = True, num_exp = None):

		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			self.config = pickle.load(config_fd)


		with open(os.path.join(model_dir, 'config_unet.pkl'), 'rb') as config_fd:
			self.config_unet = pickle.load(config_fd)


		if num_exp != None:
			self.config['num_exp'] = num_exp
			path_config_unet = join(EXP_PATH, num_exp,'config','config_unet.jason')
			path_config = join(EXP_PATH, num_exp,'config','config.jason')
			jason_dump(self.config, path_config)
			jason_dump(self.config_unet, path_config_unet)


		if unetlatent:
			self.unet_latent_model = UnetLatentModel(self.config, self.config_unet)
			self.unet_latent_model.load_state_dict(torch.load(os.path.join(model_dir, 'unetlatent.pth')))

		if ulord:
			self.ulord_model = ULordModel(self.config, self.config_unet)
			self.ulord_model .load_state_dict(torch.load(os.path.join(model_dir, 'ulord.pth')))

	def save(self, model_dir, unetlatent = True, ulord = True):

		with open(os.path.join(model_dir, 'config_unet.pkl'), 'wb') as config_fd:
			pickle.dump(self.config_unet, config_fd)

		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)


		if unetlatent:
			print("saving model......................................................")
			torch.save(self.unet_latent_model.state_dict(), os.path.join(model_dir, 'unetlatent.pth'))

		if ulord:
			print("saving model......................................................")
			torch.save(self.ulord_model.state_dict(), os.path.join(model_dir, 'ulord.pth'))

	def train_ULordModel(self, imgs ,segs, classes, imgs_u, segs_u, classes_u, model_dir, tensorboard_dir, loaded_model):
		model_name = parts(tensorboard_dir)[-1]
		path_result_train = join(EXP_PATH, self.config['num_exp'], 'results', model_name, 'train')
		path_result_val = join(EXP_PATH, self.config['num_exp'], 'results', model_name, 'val')
		path_result_loss = join(EXP_PATH, self.config['num_exp'], 'results', model_name, 'loss')

		# define logger
		logname_dice = join(EXP_PATH, self.config['num_exp'], 'logging', 'uLord_dice.log')
		logname_dice_with_a = join(EXP_PATH, self.config['num_exp'], 'logging', 'uLord_dice_witha.log')
		logname_recon = join(EXP_PATH, self.config['num_exp'], 'logging', 'uLord_recon.log')
		logname_recon_with_a = join(EXP_PATH, self.config['num_exp'], 'logging', 'uLord_recon_witha.log')

		logger_dice = define_logger('loger_dice', logname_dice)
		logger_dice_with_a = define_logger('loger_dice_witha', logname_dice_with_a )
		logger_recon = define_logger('loger_recon', logname_recon)
		logger_recon_with_a = define_logger('loger_recon_witha', logname_recon_with_a)

		if not loaded_model:
			self.ulord_model = ULordModel(self.config, self.config_unet)

		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			seg=torch.from_numpy(segs).permute(0, 3, 1, 2),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		data_u = dict(
			img = torch.from_numpy(imgs_u).permute(0, 3, 1, 2),
			seg = torch.from_numpy(segs_u).permute(0, 3, 1, 2),
			class_id = torch.from_numpy(classes_u.astype(np.int64))
		)

		rotation_transform = MyRotationTransform(angles=[-90, 0, 90])
		dataset = NamedTensorDataset(data, transform = rotation_transform  )
		dataset_u = NamedTensorDataset(data_u, transform = rotation_transform  )
		dataset_v = NamedTensorDataset(data_u)
		sampler_l = ImbalancedDatasetSampler( dataset,classes ,percent_list = self.config['train']['percent_list'])
		sampler_u = ImbalancedDatasetSampler( dataset_u , classes_u,int((len(dataset)//self.config['train']['batch_size']) * self.config['train']['batch_size_u']))

		data_loader_t = DataLoader(
			dataset,  sampler = sampler_l, batch_size = self.config['train']['batch_size'],
			num_workers = 1, pin_memory = True
		)

		data_loader_u = DataLoader(
			dataset_u, batch_size = self.config['train']['batch_size_u'], sampler = sampler_u,
			num_workers = 1, pin_memory=True
		)

		data_loader_val = DataLoader(
			dataset_v, batch_size = self.config['train']['batch_size'],
			shuffle = True, sampler = None, batch_sampler = None,
			num_workers = 10, pin_memory = True, drop_last = True
		)



		self.ulord_model.init()
		self.ulord_model.to(self.device)

		# criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)
		criterion = nn.MSELoss()
		optimizer = Adam([
			{
				'params': itertools.chain(self.ulord_model.modulation.parameters(), self.ulord_model.generator_antomy.parameters() ,self.ulord_model.segmentor.parameters(), self.ulord_model.generator.parameters()),
				'lr': self.config['train']['learning_rate']['generator']
			},
			{
				'params': itertools.chain(self.ulord_model.class_embedding.parameters()),
				'lr': self.config['train']['learning_rate']['latent']
			}
		], betas=(0.5, 0.999))

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max= self.config['train']['n_epochs'] * len(data_loader_t),
			eta_min=self.config['train']['learning_rate']['min']
		)

		summary = SummaryWriter(log_dir=tensorboard_dir)

		# train dice loss
		train_loss = AverageMeter()
		dice_loss_epoch_train = AverageMeter()
		dice_loss_epoch_val = AverageMeter()
		dice_loss_epoch_witha = AverageMeter()
		reco_loss_epoch = AverageMeter()
		reco_loss_epoch_witha= AverageMeter()


		# dice loss
		dice_loss_train = list()
		dice_loss_val = list()
		loss_list = list()

		class_decay = 0
		if self.config['Regularized_class']:
			print("here reg class")
			class_decay = self.config['class_decay']

		content_decay = 0
		if self.config['Regularized_content']:
			content_decay = self.config['content_decay']


		seg_decay = 0
		if self.config['seg_loss']:
			print("here seg decay")
			seg_decay = self.config['seg_decay']

		recon_decay = 0

		if self.config['recon_loss']:
			print("here recon_loss")
			recon_decay = self.config['recon_decay']


		fixed_sample_img = self.generate_samples_ulord(dataset, 0, path_result_train, randomized = False)
		for epoch in range(self.config['train']['n_epochs']):
			self.ulord_model.train()
			train_loss.reset()
			dice_loss_epoch_train.reset()
			print(f"epoch{epoch}")
			pbar_t = tqdm(iterable = data_loader_t)
			pbar_u = tqdm(iterable = data_loader_u)
			pbar_val = tqdm(iterable = data_loader_val)
			for i,  batch in enumerate(zip(pbar_t, pbar_u)):
				# print(batch, 'batch')
				batch_t, batch_u = batch[0], batch[1]
				batch_t = {name: tensor.to(self.device) for name, tensor in batch_t.items()}
				batch_u = {name: tensor.to(self.device) for name, tensor in batch_u.items()}
				optimizer.zero_grad()
				# print(batch['img_id'], batch['class_id'], "class id")
				out = self.ulord_model(batch_t['img'], batch_t['class_id'], batch_u['img'], batch_u['class_id'])

				class_penalty = torch.sum(out['class_code'] ** 2, dim=1).mean()
				content_penalty = torch.sum(out['content_code'] ** 2, dim=1).mean()

				reco_loss = criterion(out['img'], batch_t['img']) + criterion(out['img_u'], batch_u['img'])
				reco_loss_decay = recon_decay * reco_loss
				dice_loss_e = dice_loss(out['mask'], batch_t['seg'])
				dice_loss_d = seg_decay * dice_loss_e

				loss = reco_loss_decay + class_decay * class_penalty + dice_loss_d + content_penalty * content_decay

				loss.backward()
				optimizer.step()
				scheduler.step()

				train_loss.update(loss.item())
				dice_loss_epoch_train.update(dice_loss_e.item())
				dice_loss_epoch_witha.update(dice_loss_d.item())
				reco_loss_epoch.update(reco_loss.item())
				reco_loss_epoch_witha.update(reco_loss_decay.item())
				pbar_t.set_description_str('epoch #{}'.format(epoch))
				pbar_t.set_postfix(loss = train_loss.avg)

			pbar_t.close()
			pbar_u.close()

			logger_dice.info(f'epoch {epoch} # dice {dice_loss_epoch_train.avg}')
			logger_dice_with_a.info(f'epoch {epoch} # dice_with_a {dice_loss_epoch_witha.avg}')
			logger_recon.info(f'epoch {epoch} # reco {reco_loss_epoch.avg}')
			logger_recon_with_a.info(f'epoch {epoch} # reco_with_a {reco_loss_epoch_witha.avg}')

			dice_loss_train.append(dice_loss_epoch_train.avg)
			loss_list.append(train_loss.avg)

			for batch_v in pbar_val:
				# print("here")
				self.ulord_model.eval()
				batch_v = {name: tensor.to(self.device) for name, tensor in batch_v.items()}
				out = self.ulord_model(batch_v ['img'], batch_v ['class_id'],batch_v ['img'], batch_v ['class_id'])
				loss_val = dice_loss(out['mask'], batch_v['seg'])
				dice_loss_epoch_val.update(loss_val.item())

			# dice_txt.write('val: ' +  f'# {dice_loss_epoch_val.avg}')
			logger_dice.info(f'val epoch {epoch} # dice {dice_loss_epoch_val.avg}')
			dice_loss_val.append(dice_loss_epoch_val.avg)

			if epoch == 0:
				os.makedirs(path_result_loss, exist_ok=True)

			if epoch % 10 == 0:
				x = np.arange(epoch + 1)
				plt.figure()
				plt.plot(x, dice_loss_train)
				plt.plot(x, dice_loss_val)
				plt.title(f'Loss vs. epochs {epoch}')
				plt.ylabel('Loss')
				plt.xlabel('Epoch')
				plt.legend(['Training', 'Validation'], loc='upper right')
				plt.savefig(join(path_result_loss, f'loss_for_epoch{epoch}.jpeg'))
				plt.figure()
				plt.plot(x, loss_list)
				plt.plot(x, dice_loss_val)
				plt.title(f'Loss vs. epochs {epoch}')
				plt.ylabel('Loss')
				plt.xlabel('Epoch')
				plt.legend(['Training', 'Validation'], loc='upper right')
				plt.savefig(join(path_result_loss, f'loss_for_epoch_allloss{epoch}.jpeg'))
				print('done')




			self.save(model_dir, unetlatent = False, ulord = True)
			summary.add_scalar(tag='loss', scalar_value=train_loss.avg, global_step=epoch)

			fixed_sample_img = self.generate_samples_ulord(dataset, epoch,path_result_train,  randomized=False)
			random_sample_img = self.generate_samples_ulord(dataset, epoch,path_result_train, randomized=True)

			fixed_sample_img = self.generate_samples_ulord(dataset_u, epoch, path_result_val, randomized=False)
			random_sample_img = self.generate_samples_ulord(dataset_u, epoch, path_result_val, randomized=True)

			# summary.add_image(tag='sample-fixed', img_tensor=fixed_sample_img, global_step = epoch)
			# summary.add_image(tag='sample-random', img_tensor=random_sample_img, global_step = epoch)

		summary.close()



	def train_UnetLatentModel(self, imgs ,segs, classes, imgs_val , segs_val, classes_val , model_dir, tensorboard_dir, loaded_model):
		model_name = parts(tensorboard_dir)[-1]
		path_result_train = join(EXP_PATH, self.config['num_exp'], 'results', model_name, 'train')
		path_result_val = join(EXP_PATH, self.config['num_exp'], 'results', model_name, 'val')

		if not loaded_model:
			self.unet_latent_model = UnetLatentModel(self.config, self.config_unet)

		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			seg=torch.from_numpy(segs).permute(0, 3, 1, 2),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		data_v = dict(
			img = torch.from_numpy(imgs_val).permute(0, 3, 1, 2),
			seg = torch.from_numpy(segs_val).permute(0, 3, 1, 2),
			class_id = torch.from_numpy(classes_val.astype(np.int64))
		)


		dataset = NamedTensorDataset(data)
		dataset_val = NamedTensorDataset(data_v)

		data_loader = DataLoader(
			dataset, batch_size = self.config['train']['batch_size'],
			shuffle = True, sampler = None, batch_sampler = None,
			num_workers = 10, pin_memory = True, drop_last = True
		)




		self.unet_latent_model.init()
		self.unet_latent_model.to(self.device)

		# criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)
		criterion = nn.MSELoss()
		optimizer = Adam([
			{
				'params': itertools.chain(self.unet_latent_model.modulation.parameters(), self.unet_latent_model.generator_antomy.parameters() , self.unet_latent_model.segmentor.parameters(), self.unet_latent_model.generator.parameters()),
				'lr': self.config['train']['learning_rate']['generator']
			},
			{
				'params': itertools.chain(self.unet_latent_model.class_embedding.parameters()),
				'lr': self.config['train']['learning_rate']['latent']
			}
		], betas=(0.5, 0.999))

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max= self.config['train']['n_epochs'] * len(data_loader),
			eta_min=self.config['train']['learning_rate']['min']
		)

		summary = SummaryWriter(log_dir=tensorboard_dir)

		train_loss = AverageMeter()

		class_decay = 0
		if self.config['Regularized_class']:
			print("here reg class")
			class_decay = self.config['class_decay']

		content_decay = 0
		if self.config['Regularized_content']:
			content_decay = self.config['content_decay']


		seg_decay = 0
		if self.config['seg_loss']:
			print("here seg decay")
			seg_decay = self.config['seg_decay']

		recon_decay = 0

		if self.config['recon_loss']:
			print("here recon_loss")
			recon_decay = self.config['recon_decay']

		loss_txt = open(join(EXP_PATH,self.config['num_exp'],'logging', 'loss.txt'), 'w')
		fixed_sample_img = self.generate_samples(dataset, 0, model_name, randomized=False)
		for epoch in range(self.config['train']['n_epochs']):
			self.unet_latent_model.train()
			train_loss.reset()
			print(f"epoch{epoch}")
			pbar = tqdm(iterable = data_loader)
			for batch in pbar:

				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
				optimizer.zero_grad()
				# print(batch['img_id'], batch['class_id'], "class id")
				out = self.unet_latent_model(batch['img'], batch['class_id'])
				# content_penalty = torch.sum(out['content_code'] ** 2, dim=1).mean()
				class_penalty = torch.sum(out['class_code'] ** 2, dim=1).mean()
				content_penalty = torch.sum(out['content_code'] ** 2, dim=1).mean()

				loss = recon_decay * criterion(out['img'], batch['img']) +  class_decay * class_penalty + seg_decay * dice_loss(out['mask'], batch['seg']) + content_penalty * content_decay
				print(seg_decay * dice_loss(out['mask'], batch['seg']), "seg_decay * dice_loss(out['mask'], batch['seg'])")
				print(recon_decay * criterion(out['img'], batch['img']), "recon_decay * criterion(out['img']")
				print(content_penalty * content_decay, content_penalty, "content_penalty")
				print(class_decay * class_penalty,class_penalty, "class_penalty")

				loss.backward()
				optimizer.step()
				scheduler.step()

				train_loss.update(loss.item())
				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(loss=train_loss.avg)

			pbar.close()
			if epoch == 0:
				loss_txt.write(f'epoch {epoch} # {train_loss.avg}')
			else:
				loss_txt.write('\n' + f'epoch {epoch} # {train_loss.avg}')

			self.save(model_dir)
			summary.add_scalar(tag='loss', scalar_value=train_loss.avg, global_step=epoch)

			fixed_sample_img = self.generate_samples(dataset, epoch,path_result_train,  randomized=False)
			random_sample_img = self.generate_samples(dataset, epoch,path_result_train, randomized=True)

			fixed_sample_img = self.generate_samples(dataset_val, epoch, path_result_val, randomized=False)
			random_sample_img = self.generate_samples(dataset_val, epoch, path_result_val, randomized=True)

			# summary.add_image(tag='sample-fixed', img_tensor=fixed_sample_img, global_step = epoch)
			# summary.add_image(tag='sample-random', img_tensor=random_sample_img, global_step = epoch)

		summary.close()




	def generate_samples(self, dataset, epoch ,path_result,  n_samples=5, randomized = False):

		self.unet_latent_model.eval()
		self.seq_classifier.to(self.device)
		self.seq_classifier.eval()
		if randomized:
			random = np.random
		else:
			random = np.random.RandomState(seed = 1234)

		img_idx = torch.from_numpy(random.choice(len(dataset), size = n_samples, replace = False))
		print(img_idx, "img_idx")
		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		path_results_output = join(path_result, 'output')
		path_results_anatomy = join(path_result, 'anatomy')
		if epoch == 0:
			os.makedirs(path_results_output, exist_ok = True)
			os.makedirs(path_results_anatomy, exist_ok = True)

		cloned_all_samples = torch.clone(samples['img'])
		blank = torch.ones_like(samples['img'][0])
		blank[:,:] = 0.5

		# build antompy image
		output_antomy = list()

		for i in range(n_samples):
			if samples['class_id'][[i]]:
				class_id = torch.ones_like(samples['img'][0])
			else:
				class_id = torch.zeros_like(samples['img'][0])

			converted_imgs = [class_id]
			converted_imgs.append(samples['img'][[i]][0])
			out = self.unet_latent_model(samples['img'][[i]], samples['class_id'][[i]])
			for j in range(self.config_unet['out_channels']):
				# print(out['anatomy_img'].shape, "out['mask'][j]")
				anatomy_img = torch.unsqueeze(out['anatomy_img'][0][j], 0)
				converted_imgs.append(anatomy_img)
				# print(converted_imgs, "converted ")

			output_antomy.append(torch.cat(converted_imgs, dim = 2))

		output1 = [blank]
		classes = np.zeros((n_samples))
		for k in range(n_samples):
			if samples['class_id'][[k]]:
				classes[k] = 1
				class_id = torch.ones_like(samples['img'][0])
			else:
				class_id = torch.zeros_like(samples['img'][0])

			output1.append(class_id)

		output = [torch.cat(output1, dim=2)]
		output.append(torch.cat([blank] + list(cloned_all_samples), dim=2))
		if self.config['recon_loss']:
			print("recon here")
			class_0 = torch.tensor([0]).to(self.device)
			class_0_img = torch.zeros_like(samples['img'][0])
			converted_imgs = [class_0_img]
			torch_2 = torch.zeros_like(class_0)
			torch_2[:] = 2
			for i in range(n_samples):
				print(samples['img'][[i]].size(), "samples['img'][[i]]")
				# class_0 = torch.divide(class_0, torch_2).to(self.device)
				out = self.unet_latent_model(samples['img'][[i]], class_0)
				converted_imgs.append(out['img'][0])

			output.append(torch.cat(converted_imgs, dim=2))

			# init class 1
			class_1 = torch.tensor([1]).to(self.device)
			class_1_img = torch.ones_like(samples['img'][0])
			converted_imgs = [class_1_img]
			for i in range(n_samples):
				# class_1 = torch.divide(class_1, torch_2).to(self.device)
				out = self.unet_latent_model(samples['img'][[i]], class_1 )
				converted_imgs.append(out['img'][0])

			output.append(torch.cat(converted_imgs, dim=2))

			# classifier 0
			class_0 = torch.tensor([0]).to(self.device)
			class_0_img = torch.zeros_like(samples['img'][0])
			converted_imgs = [class_0_img]
			for i in range(n_samples):
				out = self.unet_latent_model(samples['img'][[i]], class_0)
				inp_classifier = torch.unsqueeze(out['img'][0], 0).to(self.device)
				result = self.seq_classifier(inp_classifier)
				result_img = torch.zeros_like(samples['img'][0])
				result_img[:, :] = result
				converted_imgs.append(result_img)

			output.append(torch.cat(converted_imgs, dim=2))

			# init classifier for 1
			class_1 = torch.tensor([1]).to(self.device)
			class_1_img = torch.ones_like(samples['img'][0])
			converted_imgs = [class_1_img]
			for i in range(n_samples):
				out = self.unet_latent_model(samples['img'][[i]], class_1)
				inp_classifier = torch.unsqueeze(out['img'][0], 0).to(self.device)
				result = self.seq_classifier(inp_classifier)
				result_img = torch.zeros_like(samples['img'][0])
				result_img[:, :] = result
				converted_imgs.append(result_img)

			output.append(torch.cat(converted_imgs, dim=2))

			converted_imgs = [blank]
			for i in range(n_samples):
				inp_classifier = torch.unsqueeze(samples['img'][[i]], 0).to(self.device)
				# inp_classifier = torch.tensor(samples['imgs'][[i]]).to(self.device)
				result = self.seq_classifier(inp_classifier)
				result_img = torch.zeros_like(samples['img'][0])
				result_img[:, :] = result
				converted_imgs.append(result_img)

			output.append(torch.cat(converted_imgs, dim=2))

		if self.config['seg_loss']:
			print("seg here")
			cloned_all_samples_seg = torch.clone(samples['seg'])
			output.append(torch.cat([blank] + list(cloned_all_samples_seg), dim=2))

			converted_imgs = [blank]
			for i in range(n_samples):
				out =  self.unet_latent_model(samples['img'][[i]], samples['class_id'][[i]])
				mask = torch.round(out['mask'][0])
				converted_imgs.append(mask)

			output.append(torch.cat(converted_imgs, dim=2))

			converted_imgs = [blank]
			for i in range(n_samples):
				out = self.unet_latent_model(samples['img'][[i]], samples['class_id'][[i]])
				mask = out['mask'][0]
				converted_imgs.append(mask)

			output.append(torch.cat(converted_imgs, dim=2))

		if randomized:
			rnd = 'random'
		else:
			rnd = 'not_random'

		if epoch % 10 == 0:
			output_img = torch.cat(output, dim = 1).cpu().detach().numpy()
			output_img = np.squeeze(output_img)
			merged_img = Image.fromarray((output_img * 255).astype(np.uint8))

			path_image = join(path_results_output, f' results_classes: {classes}' + '_' + f'epoch{epoch}' + '_' + rnd +'.png')
			plt.imshow(merged_img, cmap='gray')
			plt.savefig(path_image)
			# plt.show()
		if epoch % 10 == 0:
			output_img = torch.cat(output_antomy, dim = 1).cpu().detach().numpy()
			output_img = np.squeeze(output_img)
			merged_img = Image.fromarray((output_img * 255).astype(np.uint8))

			path_image = join(path_results_anatomy, f' results_classes: {classes}' + '_' + f'epoch{epoch}' + '_' + rnd +'.png')
			plt.imshow(merged_img, cmap='gray')
			plt.savefig(path_image)
			# plt.show()






	def generate_samples_ulord(self, dataset, epoch ,path_result,  n_samples=5, randomized = False):

		self.ulord_model.eval()
		if randomized:
			random = np.random
		else:
			random = np.random.RandomState(seed = 1234)

		img_idx = torch.from_numpy(random.choice(len(dataset), size = n_samples, replace = False))
		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		path_results_output = join(path_result, 'output')
		path_results_anatomy = join(path_result, 'anatomy')
		path_results_dice = join(path_result, 'dice')

		output_dice = list()
		output_dice_th = list()
		o_dice_n = list()

		if epoch == 0:
			os.makedirs(path_results_output, exist_ok = True)
			os.makedirs(path_results_anatomy, exist_ok = True)
			os.makedirs(path_results_dice, exist_ok = True)

		cloned_all_samples = torch.clone(samples['img'])
		blank = torch.ones_like(samples['img'][0])
		blank[:,:] = 0.5

		# build antompy image
		output_antomy = list()

		for i in range(n_samples):
			if samples['class_id'][[i]]:
				class_id = torch.ones_like(samples['img'][0])
			else:
				class_id = torch.zeros_like(samples['img'][0])

			converted_imgs = [class_id]
			converted_imgs.append(samples['img'][[i]][0])
			out = self.ulord_model(samples['img'][[i]], samples['class_id'][[i]], samples['img'][[i]], samples['class_id'][[i]])
			for j in range(self.config_unet['out_channels']):

				anatomy_img = torch.unsqueeze(out['anatomy_img'][0][j], 0)
				converted_imgs.append(anatomy_img)


			output_antomy.append(torch.cat(converted_imgs, dim = 2))

		output1 = [blank]
		classes = np.zeros((n_samples))
		for k in range(n_samples):
			if samples['class_id'][[k]]:
				classes[k] = 1
				class_id = torch.ones_like(samples['img'][0])
			else:
				class_id = torch.zeros_like(samples['img'][0])

			output1.append(class_id)

		output = [torch.cat(output1, dim=2)]
		output.append(torch.cat([blank] + list(cloned_all_samples), dim=2))
		if self.config['recon_loss']:
			# init class 1
			class_0 = torch.tensor([0]).to(self.device)
			class_0_img = torch.zeros_like(samples['img'][0])
			converted_imgs = [class_0_img]
			torch_2 = torch.zeros_like(class_0)
			torch_2[:] = 2
			for i in range(n_samples):
				# print(samples['img'][[i]].size(), "samples['img'][[i]]")
				# class_0 = torch.divide(class_0, torch_2).to(self.device)
				out = self.ulord_model(samples['img'][[i]], class_0, samples['img'][[i]], class_0)
				converted_imgs.append(out['img'][0])

			output.append(torch.cat(converted_imgs, dim=2))

			# init class 1
			class_1 = torch.tensor([1]).to(self.device)
			class_1_img = torch.ones_like(samples['img'][0])
			converted_imgs = [class_1_img]
			for i in range(n_samples):
				# class_1 = torch.divide(class_1, torch_2).to(self.device)
				out = self.ulord_model(samples['img'][[i]], class_1, samples['img'][[i]], class_1)
				converted_imgs.append(out['img'][0])

			output.append(torch.cat(converted_imgs, dim=2))



		if self.config['seg_loss']:

			cloned_all_samples_seg = torch.clone(samples['seg'])
			output.append(torch.cat([blank] + list(cloned_all_samples_seg), dim=2))

			converted_imgs = [blank]

			for i in range(n_samples):
				out =  self.ulord_model(samples['img'][[i]], samples['class_id'][[i]], samples['img'][[i]], samples['class_id'][[i]])
				mask = torch.round(out['mask'][0])
				converted_imgs.append(mask)
				output_dice.append(dice_loss(mask,samples['seg'][[i]]))


			output.append(torch.cat(converted_imgs, dim=2))

			converted_imgs = [blank]
			for i in range(n_samples):
				out = self.ulord_model(samples['img'][[i]], samples['class_id'][[i]], samples['img'][[i]], samples['class_id'][[i]])
				mask = out['mask'][0]

				converted_imgs.append(mask)
				output_dice_th.append(dice_loss(mask,samples['seg'][[i]]))

			output.append(torch.cat(converted_imgs, dim=2))

		if randomized:
			rnd = 'random'
		else:
			rnd = 'not_random'

		if epoch % 10 == 0:

			output_img = torch.cat(output, dim = 1).cpu().detach().numpy()
			output_img = np.squeeze(output_img)
			merged_img = Image.fromarray((output_img * 255).astype(np.uint8))
			plt.figure()
			path_image = join(path_results_output, f' results_classes: {classes}' + '_' + f'epoch{epoch}' + '_' + rnd +'.png')
			plt.title(f"epoch number: {epoch}")
			plt.imshow(merged_img, cmap='gray')
			plt.savefig(path_image)
			# plt.show()
		if epoch % 10 == 0:
			output_img = torch.cat(output_antomy, dim = 1).cpu().detach().numpy()
			output_img = np.squeeze(output_img)
			merged_img = Image.fromarray((output_img * 255).astype(np.uint8))
			plt.figure()
			path_image = join(path_results_anatomy, f' results_classes: {classes}' + '_' + f'epoch{epoch}' + '_' + rnd +'.png')
			plt.title(f"epoch number: {epoch}")
			plt.imshow(merged_img, cmap='gray')
			plt.savefig(path_image)
			# plt.show()
		if epoch % 10 == 0:
			output_dice = torch.tensor(output_dice).cpu().detach().numpy()
			x = np.arange(n_samples)
			path_image = join(path_results_dice,
							  f' results_classes: {classes}' + '_' + f'epoch{epoch}' + '_' + rnd + '.png')
			plt.figure()
			plt.title(f"epoch number: {epoch}")
			plt.bar(x, output_dice)
			plt.savefig(path_image)



	def generate_samples_amortized(self, dataset, n_samples=5, randomized=False):
		self.amortized_model.eval()

		if randomized:
			random = np.random
		else:
			random = np.random.RandomState(seed=1234)

		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))

		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

		blank = torch.ones_like(samples['img'][0])
		output = [torch.cat([blank] + list(samples['img']), dim=2)]
		for i in range(n_samples):
			converted_imgs = [samples['img'][i]]

			for j in range(n_samples):
				out = self.amortized_model.convert(samples['img'][[j]], samples['img'][[i]])
				converted_imgs.append(out['img'][0])

			output.append(torch.cat(converted_imgs, dim=2))

		return torch.cat(output, dim=1)
