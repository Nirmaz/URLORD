import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import pickle
from torch.autograd import Function
from model.unet2d_model import UNet, Segmentntor
import numpy as np
import time

CUDA_LAUNCH_BLOCKING=1

def pickle_load(in_file):
	with open(in_file, "rb") as opened_file:
		return pickle.load(opened_file)

class my_round_func(Function):

	@staticmethod
	def forward(ctx, input):
		ctx.input = input
		return torch.round(input)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output.clone()
		return grad_input

class ClassEmbeddingFromSource(nn.Module):
	"""
	load class embedding from specific source
	"""

	def __init__(self, class_embedding):
		super().__init__()

		self.class_embedding_tesnor = class_embedding

	def forward(self, x):
		x = self.class_embedding_tesnor[x]
		return x

def conv_layer(dim: int):
	if dim == 3:
		return nn.Conv3d
	elif dim == 2:
		return nn.Conv2d


def get_conv_layer(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = True, dim: int = 2):
	return conv_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)



class UNetC(nn.Module):

	def __init__(self, config_unet):
		super().__init__()

		self.config_unet = config_unet
		self.unet = UNet(self.config_unet)
		self.act = nn.ReLU()
		self.last_conv = get_conv_layer(self.config_unet['out_channels'], 1, kernel_size = 3, stride = 1, padding = 1,
									bias = True, dim=config_unet['dim'])
		self.last_act = nn.Sigmoid()
	def forward(self, img):

		x = self.unet(img)
		x = self.act(x)
		x = self.last_conv(x)
		x = self.last_act(x)

		# print("--- %s unetd time ---" % (time.time() - start_time))
		return {
			'mask': x
		}

	def init(self):
		self.apply(self.weights_init)

	@staticmethod
	def weights_init(m):
		if isinstance(m, nn.Embedding):
			nn.init.uniform_(m.weight, a=-0.05, b=0.05)
			# nn.init.uniform_(m.weight, a = 0.45, b = 0.55)





class ULordSModel(nn.Module):

	def __init__(self, config, config_unet):
		super().__init__()

		self.config_unet = config_unet
		self.config = config
		if config['Regularized_class']:
			self.class_embedding = RegularizedEmbedding(config['n_classes'], config['class_dim'], config['class_std'])
		else:
			self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])

		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
		self.generator_antomy = UNet(self.config_unet)

		self.generator = Generator(config['n_adain_layers'], config['adain_dim'], config['img_shape'], self.config_unet['out_channels'])
		# self.round_act = nn.ReLU()
		self.rounding = Rounding()
		self.flatten = nn.Flatten()

	def forward(self, img, class_id, img_u, class_id_u):


		anatomy_img = self.generator_antomy(img)
		class_code = self.class_embedding(class_id)

		anatomy_img_u = self.generator_antomy(img_u)
		class_code_u = self.class_embedding(class_id_u)

		# print(class_id, "class_id,")
		if self.config['round']:
			# print(anatomy_img[anatomy_img > 0.5], "before")
			idx = anatomy_img > 0.5
			anatomy_img = self.rounding(anatomy_img)
			anatomy_img_u = self.rounding(anatomy_img_u)

		# get the full image
		content_code = self.flatten(anatomy_img)
		content_code_u = self.flatten(anatomy_img_u)

		class_adain_params = self.modulation(class_code)
		class_adain_params_u = self.modulation(class_code_u)

		generated_img = self.generator(anatomy_img, class_adain_params)
		generated_img_u = self.generator(anatomy_img_u, class_adain_params_u)

		return {
			'img': generated_img,
			'img_u': generated_img_u,
			'anatomy_img': anatomy_img,
			'anatomy_img_u': anatomy_img_u,
			'class_code': class_code,
			'class_code_u': class_code_u,
			'content_code': content_code,
			'content_code_u': content_code_u
		}

	def init(self):
		self.apply(self.weights_init)

	@staticmethod
	def weights_init(m):
		if isinstance(m, nn.Embedding):
			# nn.init.uniform_(m.weight, a=-0.05, b=0.05)
			nn.init.uniform_(m.weight, a = 0.45, b = 0.55)





class ULordModel(nn.Module):

	def __init__(self, config, config_unet):
		super().__init__()

		self.config_unet = config_unet
		self.config = config
		if config['Regularized_class']:
			self.class_embedding = RegularizedEmbedding(config['n_classes'], config['class_dim'], config['class_std'])
		else:
			self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])

		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
		self.generator_antomy = UNet(self.config_unet)
		self.segmentor = Segmentntor(self.config_unet)
		self.generator = Generator(config['n_adain_layers'], config['adain_dim'], config['img_shape'][2], self.config_unet['out_channels'])
		# self.round_act = nn.ReLU()
		self.rounding = Rounding()
		self.flatten = nn.Flatten()

	def forward(self, img, class_id, img_u, class_id_u):
		# print(img.size(), "img size,")
		# print(class_id.size(), "size id labaled")
		# print(class_id_u.size(), "class_id_u size")
		# print(torch.count_nonzero(class_id), "class id none zero")
		# print(torch.count_nonzero(class_id_u), "class id u none zero")
		# print(img_u.size(), "img_u.size()")
		anatomy_img = self.generator_antomy(img)
		class_code = self.class_embedding(class_id)

		anatomy_img_u = self.generator_antomy(img_u)
		class_code_u = self.class_embedding(class_id_u)

		# print(class_id, "class_id,")
		if self.config['round']:
			# print(anatomy_img[anatomy_img > 0.5], "before")
			idx = anatomy_img > 0.5
			anatomy_img = self.rounding(anatomy_img)
			anatomy_img_u = self.rounding(anatomy_img_u)

		# get the full image
		content_code = self.flatten(anatomy_img)
		content_code_u = self.flatten(anatomy_img_u)

		class_adain_params = self.modulation(class_code)
		class_adain_params_u = self.modulation(class_code_u)

		generated_img = self.generator(anatomy_img, class_adain_params)
		generated_img_u = self.generator(anatomy_img_u, class_adain_params_u)

		if self.config['segmentor_gard']:
			segmentor_input = anatomy_img
			segmentor_input_u = anatomy_img_u
			# print("here")
		else:
			segmentor_input = anatomy_img.detach()
			segmentor_input_u = anatomy_img_u.detach()



		mask = self.segmentor(segmentor_input)
		mask_u = self.segmentor(segmentor_input_u)

		return {
			'img': generated_img,
			'img_u': generated_img_u,
			'anatomy_img': anatomy_img,
			'anatomy_img_u': anatomy_img_u,
			'mask': mask,
			'mask_u': mask_u,
			'class_code': class_code,
			'class_code_u': class_code_u,
			'content_code': content_code,
			'content_code_u': content_code_u
		}

	def init(self):
		self.apply(self.weights_init)

	@staticmethod
	def weights_init(m):
		if isinstance(m, nn.Embedding):
			nn.init.uniform_(m.weight, a=-0.05, b=0.05)
			# nn.init.uniform_(m.weight, a = 0.45, b = 0.55)




class ULordModel3D(nn.Module):

	def __init__(self, config, config_unet):
		super().__init__()

		self.config_unet = config_unet
		self.config = config
		if config['Regularized_class']:
			self.class_embedding = RegularizedEmbedding(config['n_classes'], config['class_dim'], config['class_std'])
		else:
			self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])

		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
		self.generator_antomy = UNet(self.config_unet)
		self.segmentor = Segmentntor(self.config_unet)
		# self.generator = Generator(config['n_adain_layers'], config['adain_dim'], 1, self.config_unet['out_channels'])
		# self.generator = Generator3D(config['n_adain_layers'], config['adain_dim'], 1, self.config_unet['out_channels'], config['film_layer'])
		self.generator = Generator3D(config['n_adain_layers'], config['adain_dim'], 1, self.config_unet['out_channels'], False)
		# self.round_act = nn.ReLU()
		self.rounding = Rounding()
		# self.rounding = my_round_func
		self.softmax = nn.Softmax(dim = 1)
		self.flatten = nn.Flatten()
		self.last_act = nn.Softmax(dim = 1)

	def forward(self, img, class_id, img_u, class_id_u):
		depth = img.size()[2]
		class_id = class_id.repeat_interleave(depth)
		class_id_u = class_id_u.repeat_interleave(depth)


		anatomy_img = self.generator_antomy(img)
		anatomy_img = self.last_act(anatomy_img)
		class_code = self.class_embedding(class_id)

		s_a_img = anatomy_img.size()

		# anatomy_img = torch.reshape(anatomy_img, (s_a_img[0] * self.config_unet['in_channels'], self.config_unet['anatomy_rep'], s_a_img[2], s_a_img[3]))
		# anatomy_img = self.softmax(anatomy_img)
		# anatomy_img = torch.reshape(anatomy_img, (s_a_img[0], s_a_img[1], s_a_img[2], s_a_img[3]))

		anatomy_img_u = self.generator_antomy(img_u)
		anatomy_img_u = self.last_act(anatomy_img_u)
		class_code_u = self.class_embedding(class_id_u)
		s_a_img_u = anatomy_img_u.size()

		# apply softmax on the z axis
		# anatomy_img_u = torch.reshape(anatomy_img_u, (s_a_img_u[0] * self.config_unet['in_channels'], self.config_unet['anatomy_rep'], s_a_img_u[2], s_a_img_u[3]))
		# anatomy_img_u = self.softmax(anatomy_img_u)
		# anatomy_img_u = torch.reshape(anatomy_img_u, (s_a_img_u[0], s_a_img_u[1], s_a_img_u[2], s_a_img_u[3]))
		#
		# anatomy_img = torch.reshape(anatomy_img, (
		# s_a_img[0] * self.config_unet['in_channels'], self.config_unet['anatomy_rep'], s_a_img[2], s_a_img[3]))
		# anatomy_img = self.softmax(anatomy_img)
		# anatomy_img = torch.reshape(anatomy_img, (s_a_img[0], s_a_img[1], s_a_img[2], s_a_img[3]))

		# apply rounding
		if self.config['round']:
			anatomy_img = self.rounding(anatomy_img)
			anatomy_img_u = self.rounding(anatomy_img_u)

		# get the full image
		content_code = self.flatten(anatomy_img)
		content_code_u = self.flatten(anatomy_img_u)

		class_adain_params = self.modulation(class_code)
		class_adain_params_u = self.modulation(class_code_u)

		# inp_gen = anatomy_img.contiguous().view(s_a_img[0] * s_a_img[2],s_a_img[1], s_a_img[3],s_a_img[4])
		# inp_gen_u = anatomy_img_u.contiguous().view(s_a_img_u[0] * s_a_img_u[2],s_a_img_u[1], s_a_img_u[3],s_a_img_u[4])
		# #
		inp_gen = anatomy_img
		inp_gen_u = anatomy_img_u

		generated_img = self.generator(inp_gen, class_adain_params)
		generated_img_u = self.generator(inp_gen_u, class_adain_params_u)


		# print(generated_img_u.size(), " generated_img usizeeeee anat")

		# generated_img = generated_img.contiguous().view(s_a_img[0], 1, s_a_img[2],s_a_img[3], s_a_img[4])
		# generated_img_u = generated_img_u.contiguous().view(s_a_img_u[0] , 1, s_a_img_u[2], s_a_img_u[3], s_a_img_u[4])

		if self.config['segmentor_gard']:
			segmentor_input = anatomy_img
			segmentor_input_u = anatomy_img_u
			# print("here")
		else:
			segmentor_input = anatomy_img.detach()
			segmentor_input_u = anatomy_img_u.detach()



		mask = self.segmentor(segmentor_input)
		mask_u = self.segmentor(segmentor_input_u)

		return {
			'img': generated_img,
			'img_u': generated_img_u,
			'anatomy_img': anatomy_img,
			'anatomy_img_u': anatomy_img_u,
			'mask': mask,
			'mask_u': mask_u,
			'class_code': class_code,
			'class_code_u': class_code_u,
			'content_code': content_code,
			'content_code_u': content_code_u
		}

	# def forward(self, img, class_id, img_u, class_id_u):
	# 	img_batch_size = img.size()[0]
	# 	img_u_batch_size = img_u.size()[0]
	# 	# start_time0 = time.time()
	# 	depth = img.size()[2]
	# 	class_id = class_id.repeat_interleave(depth)
	# 	class_id_u = class_id_u.repeat_interleave(depth)
	# 	# anatomy_img = self.generator_antomy(img)
	# 	class_code = self.class_embedding(class_id)
	#
	# 	# anatomy_img_u = self.generator_antomy(img_u)
	# 	class_code_u = self.class_embedding(class_id_u)
	# 	# start_time0_1 = time.time()
	# 	anatomy = self.generator_antomy(torch.cat([img, img_u], dim=0))
	# 	anatomy_img, anatomy_img_u = anatomy[:img_batch_size, :], anatomy[img_batch_size: img_batch_size + img_u_batch_size  , :]
	#
	# 	# s_a_img_u = anatomy_img_u.size()
	# 	# apply softmax on the z axis
	# 	# anatomy_img_u = torch.reshape(anatomy_img_u, (s_a_img_u[0] * self.config_unet['in_channels'], self.config_unet['anatomy_rep'], s_a_img_u[2], s_a_img_u[3]))
	# 	# anatomy_img_u = self.softmax(anatomy_img_u)
	# 	# anatomy_img_u = torch.reshape(anatomy_img_u, (s_a_img_u[0], s_a_img_u[1], s_a_img_u[2], s_a_img_u[3]))
	# 	#
	# 	# anatomy_img = torch.reshape(anatomy_img, (
	# 	# s_a_img[0] * self.config_unet['in_channels'], self.config_unet['anatomy_rep'], s_a_img[2], s_a_img[3]))
	# 	# anatomy_img = self.softmax(anatomy_img)
	# 	# anatomy_img = torch.reshape(anatomy_img, (s_a_img[0], s_a_img[1], s_a_img[2], s_a_img[3]))
	#
	# 	# apply rounding
	# 	if self.config['round']:
	#
	# 		anatomy_img = self.rounding(anatomy_img)
	# 		anatomy_img_u = self.rounding(anatomy_img_u)
	#
	# 		# anatomy_img = self.rounding.apply(anatomy_img)
	# 		# anatomy_img_u = self.rounding.apply(anatomy_img_u)
	#
	# 	# get the full image
	# 	content_code = self.flatten(anatomy_img)
	# 	content_code_u = self.flatten(anatomy_img_u)
	# 	class_adain_params = self.modulation(class_code)
	# 	class_adain_params_u = self.modulation(class_code_u)
	# 	start_time0_2 = time.time()
	# 	# print("--- %s modu time ---" % (time.time() - start_time1))
	# 	# inp_gen = anatomy_img.contiguous().view(s_a_img[0] * s_a_img[2],s_a_img[1], s_a_img[3],s_a_img[4])
	# 	# inp_gen_u = anatomy_img_u.contiguous().view(s_a_img_u[0] * s_a_img_u[2],s_a_img_u[1], s_a_img_u[3],s_a_img_u[4])
	# 	# #
	# 	inp_gen = anatomy_img
	# 	inp_gen_u = anatomy_img_u
	# 	generated_img = self.generator(inp_gen, class_adain_params)
	# 	generated_img_u = self.generator(inp_gen_u, class_adain_params_u)
	# 	# start_time0_3 = time.time()
	# 	# print("--- %s genret time ---" % (time.time() - start_time2))
	# 	# print(generated_img_u.size(), " generated_img usizeeeee anat")
	#
	# 	# generated_img = generated_img.contiguous().view(s_a_img[0], 1, s_a_img[2],s_a_img[3], s_a_img[4])
	# 	# generated_img_u = generated_img_u.contiguous().view(s_a_img_u[0] , 1, s_a_img_u[2], s_a_img_u[3], s_a_img_u[4])
	# 	# start_time3 = time.time()
	# 	if self.config['segmentor_gard']:
	# 		segmentor_input = anatomy_img
	# 		segmentor_input_u = anatomy_img_u
	# 		# print("here")
	# 	else:
	# 		segmentor_input = anatomy_img.detach()
	# 		segmentor_input_u = anatomy_img_u.detach()
	#
	# 	mask = self.segmentor(segmentor_input)
	# 	mask_u = self.segmentor(segmentor_input_u)
	# 	# print("--- %s seg time ---" % (time.time() - start_time3))
	# 	# print("--- %s seg time alll ---" % (time.time() - start_time0))
	# 	# print("--- %s seg time alll 0_1 ---" % (time.time() - start_time0_1))
	# 	# print("--- %s seg time alll 0_2 ---" % (time.time() - start_time0_2))
	# 	# print("--- %s seg time alll 0_3 ---" % (time.time() - start_time0_3))
	# 	return {
	# 		'img': generated_img,
	# 		'img_u': generated_img_u,
	# 		'anatomy_img': anatomy_img,
	# 		'anatomy_img_u': anatomy_img_u,
	# 		'mask': mask,
	# 		'mask_u': mask_u,
	# 		'class_code': class_code,
	# 		'class_code_u': class_code_u,
	# 		'content_code': content_code,
	# 		'content_code_u': content_code_u
	# 	}

	def init(self):
		self.apply(self.weights_init)

	@staticmethod
	def weights_init(m):
		if isinstance(m, nn.Embedding):
			nn.init.uniform_(m.weight, a=-0.05, b=0.05)
			# nn.init.uniform_(m.weight, a = 0.45, b = 0.55)



class UnetLatentModel(nn.Module):

	def __init__(self, config, config_unet):
		super().__init__()

		self.config_unet = config_unet
		self.config = config
		if config['Regularized_class']:
			self.class_embedding = RegularizedEmbedding(config['n_classes'], config['class_dim'], config['class_std'])
		else:
			self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])

		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
		self.generator_antomy = UNet(self.config_unet)
		self.segmentor = Segmentntor(self.config_unet)
		self.generator = Generator(config['n_adain_layers'], config['adain_dim'], config['img_shape'], self.config_unet['out_channels'])
		# self.round_act = nn.ReLU()
		self.rounding = Rounding()
		self.flatten = nn.Flatten()

	def forward(self, img, class_id):
		# print(img.size(), "img size,")
		print(class_id, "class_id,")
		anatomy_img = self.generator_antomy(img)
		class_code = self.class_embedding(class_id) * 2
		print(class_code, "class_id,")
		# print(class_id, "class_id,")
		if self.config['round']:
			# print(anatomy_img[anatomy_img > 0.5], "before")
			idx = anatomy_img > 0.5
			anatomy_img = self.rounding(anatomy_img)
			# epsilon = torch.ones_like(anatomy_img)
			# epsilon = epsilon / 10000
			# one_matrix = one_matrix / 2
			# anatomy_img_subtracted =
			# anatomy_img_subtracted_r = self.round_act(anatomy_img_subtracted)
			# m = anatomy_img_subtracted + epsilon
			# anatomy_img = torch.divide(anatomy_img_subtracted_r, m)
			# print(anatomy_img[idx], "after")
			# print(anatomy_img[0, :, 0, 0], "after_fround")

		# get the full image
		content_code = self.flatten(anatomy_img)
		class_adain_params = self.modulation(class_code)
		# print(class_adain_params,"class adain paramters" )
		# print(class_code,"class_code," )
		generated_img = self.generator(anatomy_img, class_adain_params)
		mask = self.segmentor(anatomy_img)
		return {
			'img': generated_img,
			'anatomy_img': anatomy_img,
			'mask': mask,
			'class_code': class_code,
			'content_code': content_code
		}

	def init(self):
		self.apply(self.weights_init)

	@staticmethod
	def weights_init(m):
		if isinstance(m, nn.Embedding):
			# nn.init.uniform_(m.weight, a=-0.05, b=0.05)
			nn.init.uniform_(m.weight, a = 0.45, b = 0.55)




class RegularizedEmbedding(nn.Module):

	def __init__(self, num_embeddings, embedding_dim, stddev):
		super().__init__()

		self.embedding = nn.Embedding(num_embeddings, embedding_dim)
		self.stddev = stddev

	def forward(self, x):
		x = self.embedding(x)

		if self.training and self.stddev != 0:
			noise = torch.zeros_like(x)
			noise.normal_(mean = 0, std = self.stddev)

			x = x + noise

		return x


class Rounding(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, x):

		Rounding_factor = torch.zeros_like(x.detach(), requires_grad = False)
		tensor = torch.tensor(x, requires_grad = False)
		# print(tensor.size(), "tensor")
		Rounding_factor[tensor <= 0.5] = tensor[tensor <= 0.5] * (-1)
		Rounding_factor[tensor > 0.5] = 1 - tensor[tensor > 0.5]

		x = x + Rounding_factor

		return x



class Modulation(nn.Module):

	def __init__(self, code_dim, n_adain_layers, adain_dim):
		super().__init__()

		self.__n_adain_layers = n_adain_layers
		self.__adain_dim = adain_dim

		self.adain_per_layer = nn.ModuleList([
			nn.Linear(in_features = code_dim, out_features = adain_dim * 2)
			for _ in range(n_adain_layers)
		])

	def forward(self, x):

		adain_all = torch.cat([f(x) for f in self.adain_per_layer], dim = -1)
		# print(adain_all.size(), "size(")
		adain_params = adain_all.reshape(-1, self.__n_adain_layers, self.__adain_dim, 2)
		# print(adain_params.size(), "adsin parms size(")


		return adain_params


class Generator(nn.Module):

	def __init__(self, n_adain_layers, adain_dim, out_channel, in_channel):
		super().__init__()

		self.__adain_dim = adain_dim

		self.adain_conv_layers = nn.ModuleList()
		for i in range(n_adain_layers):
			if i == 0:
				in_channels_adain = in_channel
			else:
				in_channels_adain = adain_dim

			self.adain_conv_layers += [
				nn.Conv2d(in_channels = in_channels_adain, out_channels=adain_dim, padding= 1, kernel_size=3),
				nn.LeakyReLU(),
				AdaptiveInstanceNorm2d(adain_layer_idx = i)
			]

		self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)

		self.last_conv_layers = nn.Sequential(
			nn.Conv2d(in_channels = adain_dim, out_channels = 64, padding = 2, kernel_size = 5),
			nn.LeakyReLU(),

			nn.Conv2d(in_channels = 64, out_channels = out_channel, padding = 3, kernel_size = 7),
			nn.Sigmoid()
		)

	def assign_adain_params(self, adain_params):
		for m in self.adain_conv_layers.modules():
			if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
				m.bias = adain_params[:, m.adain_layer_idx, :, 0]
				m.weight = adain_params[:, m.adain_layer_idx, :, 1]

	def forward(self, x, class_adain_params):
		self.assign_adain_params(class_adain_params)
		x = self.adain_conv_layers(x)
		x = self.last_conv_layers(x)
		return x

class Generator3D(nn.Module):

	def __init__(self, n_adain_layers, adain_dim, out_channel, in_channel, film_layer):
		super().__init__()

		self.__adain_dim = adain_dim

		self.adain_conv_layers = nn.ModuleList()
		for i in range(n_adain_layers):
			if i == 0:
				in_channels_adain = in_channel
			else:
				in_channels_adain = adain_dim

			self.adain_conv_layers += [
				nn.Conv3d(in_channels=in_channels_adain, out_channels=adain_dim, padding=1, kernel_size = 3),
				nn.LeakyReLU(),
				AdaptiveInstanceNorm3d(adain_layer_idx=i, film_layer = film_layer)
			]

		self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)

		self.last_conv_layers = nn.Sequential(
			nn.Conv3d(in_channels=adain_dim, out_channels=64, padding=2, kernel_size=5),
			nn.LeakyReLU(),

			nn.Conv3d(in_channels = 64, out_channels = out_channel, padding=3, kernel_size=7),
			nn.Sigmoid()
		)

	def assign_adain_params(self, adain_params):
		for m in self.adain_conv_layers.modules():
			if m.__class__.__name__ == "AdaptiveInstanceNorm3d":
				m.bias = adain_params[:, m.adain_layer_idx, :, 0]
				m.weight = adain_params[:, m.adain_layer_idx, :, 1]

	def forward(self, x, class_adain_params):
		self.assign_adain_params(class_adain_params)
		x = self.adain_conv_layers(x)
		x = self.last_conv_layers(x)
		return x

class Encoder(nn.Module):

	def __init__(self, img_shape, code_dim):
		super().__init__()

		self.conv_layers = nn.Sequential(
			nn.Conv2d(in_channels=img_shape[-1], out_channels=64, kernel_size=7, stride=1, padding=3),
			nn.LeakyReLU(),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(),

			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(),

			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(),

			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU()
		)

		self.fc_layers = nn.Sequential(
			nn.Linear(in_features=4096, out_features=256),
			nn.LeakyReLU(),

			nn.Linear(in_features=256, out_features=256),
			nn.LeakyReLU(),

			nn.Linear(256, code_dim)
		)

	def forward(self, x):
		batch_size = x.shape[0]
		x = self.conv_layers(x)
		x = x.contiguous().view((batch_size, -1))

		x = self.fc_layers(x)
		return x

class AdaptiveInstanceNorm3d(nn.Module):

	def __init__(self, adain_layer_idx, film_layer):
		super().__init__()
		self.weight = None
		self.bias = None
		self.adain_layer_idx = adain_layer_idx
		self.film_layer = film_layer

	def forward(self, x):
		b, c, D = x.shape[0], x.shape[1], x.shape[2]

		x_reshaped = x.contiguous().view(1, b * c * D , *x.shape[3:])
		# print(self.weight.size(), "weights size")
		weight = self.weight.contiguous().view(-1)
		bias = self.bias.contiguous().view(-1)
		if self.film_layer:
			weight = weight.unsqueeze(dim=0)
			weight = weight.unsqueeze(dim=2)
			weight = weight.unsqueeze(dim=3)

			bias = bias.unsqueeze(dim=0)
			bias = bias.unsqueeze(dim=2)
			bias = bias.unsqueeze(dim=3)

			r = torch.mul(x_reshaped, weight)
			out = torch.add(r, bias)
		else:
			out = F.batch_norm(
				x_reshaped, running_mean=None, running_var=None,
				weight=weight, bias=bias, training=True
			)


		# print(weight.size(), "weights size")
		# exit()

		out = out.view(b, c, D,  *x.shape[3:])
		return out

class AdaptiveInstanceNorm2d(nn.Module):

	def __init__(self, adain_layer_idx):
		super().__init__()
		self.weight = None
		self.bias = None
		self.adain_layer_idx = adain_layer_idx

	def forward(self, x):
		b, c = x.shape[0], x.shape[1]

		x_reshaped = x.contiguous().view(1, b * c, *x.shape[2:])
		weight = self.weight.contiguous().view(-1)
		bias = self.bias.contiguous().view(-1)

		out = F.batch_norm(
			x_reshaped, running_mean=None, running_var=None,
			weight=weight, bias=bias, training=True
		)

		out = out.view(b, c, *x.shape[2:])
		return out

class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)

class NetVGGFeatures(nn.Module):

	def __init__(self, layer_ids):
		super().__init__()

		self.vggnet = models.vgg16(pretrained = True)
		self.layer_ids = layer_ids

	def forward(self, img):
		output = []

		if img.shape[1] == 1:
			x = LambdaLayer(lambda t: torch.repeat_interleave(t, 3, 1))(img)
		else:
			x = img

		for i in range(self.layer_ids[-1] + 1):
			x = self.vggnet.features[i](x)

			if i in self.layer_ids:
				output.append(x)

		return output


class VGGDistance(nn.Module):

	def __init__(self, layer_ids):
		super().__init__()

		self.vgg = NetVGGFeatures(layer_ids)
		self.layer_ids = layer_ids

	def forward(self, I1, I2):
		b_sz = I1.size(0)
		f1 = self.vgg(I1)
		f2 = self.vgg(I2)

		loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)

		for i in range(len(self.layer_ids)):
			layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
			loss = loss + layer_loss

		return loss.mean()
# =================================old version ===============================
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torchvision import models
# import pickle
# from model.unet2d_model import UNet, Segmentntor
# import numpy as np
# CUDA_LAUNCH_BLOCKING=1
#
# def pickle_load(in_file):
# 	with open(in_file, "rb") as opened_file:
# 		return pickle.load(opened_file)
#
#
# class ClassEmbeddingFromSource(nn.Module):
# 	"""
# 	load class embedding from specific source
# 	"""
#
# 	def __init__(self, class_embedding):
# 		super().__init__()
#
# 		self.class_embedding_tesnor = class_embedding
#
# 	def forward(self, x):
# 		x = self.class_embedding_tesnor[x]
# 		return x
#
# class UNetC(nn.Module):
#
# 	def __init__(self, config_unet):
# 		super().__init__()
#
# 		self.config_unet = config_unet
# 		self.unet = UNet(self.config_unet)
#
# 	def forward(self, img):
# 		x = self.unet(img)
# 		return {
# 			'mask': x
# 		}
#
# 	def init(self):
# 		self.apply(self.weights_init)
#
# 	@staticmethod
# 	def weights_init(m):
# 		if isinstance(m, nn.Embedding):
# 			nn.init.uniform_(m.weight, a=-0.05, b=0.05)
# 			# nn.init.uniform_(m.weight, a = 0.45, b = 0.55)
#
#
#
#
#
# class ULordSModel(nn.Module):
#
# 	def __init__(self, config, config_unet):
# 		super().__init__()
#
# 		self.config_unet = config_unet
# 		self.config = config
# 		if config['Regularized_class']:
# 			self.class_embedding = RegularizedEmbedding(config['n_classes'], config['class_dim'], config['class_std'])
# 		else:
# 			self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
#
# 		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
# 		self.generator_antomy = UNet(self.config_unet)
#
# 		self.generator = Generator(config['n_adain_layers'], config['adain_dim'], config['img_shape'], self.config_unet['out_channels'])
# 		# self.round_act = nn.ReLU()
# 		self.rounding = Rounding()
# 		self.flatten = nn.Flatten()
#
# 	def forward(self, img, class_id, img_u, class_id_u):
#
#
# 		anatomy_img = self.generator_antomy(img)
# 		class_code = self.class_embedding(class_id)
#
# 		anatomy_img_u = self.generator_antomy(img_u)
# 		class_code_u = self.class_embedding(class_id_u)
#
# 		# print(class_id, "class_id,")
# 		if self.config['round']:
# 			# print(anatomy_img[anatomy_img > 0.5], "before")
# 			idx = anatomy_img > 0.5
# 			anatomy_img = self.rounding(anatomy_img)
# 			anatomy_img_u = self.rounding(anatomy_img_u)
#
# 		# get the full image
# 		content_code = self.flatten(anatomy_img)
# 		content_code_u = self.flatten(anatomy_img_u)
#
# 		class_adain_params = self.modulation(class_code)
# 		class_adain_params_u = self.modulation(class_code_u)
#
# 		generated_img = self.generator(anatomy_img, class_adain_params)
# 		generated_img_u = self.generator(anatomy_img_u, class_adain_params_u)
#
# 		return {
# 			'img': generated_img,
# 			'img_u': generated_img_u,
# 			'anatomy_img': anatomy_img,
# 			'anatomy_img_u': anatomy_img_u,
# 			'class_code': class_code,
# 			'class_code_u': class_code_u,
# 			'content_code': content_code,
# 			'content_code_u': content_code_u
# 		}
#
# 	def init(self):
# 		self.apply(self.weights_init)
#
# 	@staticmethod
# 	def weights_init(m):
# 		if isinstance(m, nn.Embedding):
# 			# nn.init.uniform_(m.weight, a=-0.05, b=0.05)
# 			nn.init.uniform_(m.weight, a = 0.45, b = 0.55)
#
#
#
#
#
# class ULordModel(nn.Module):
#
# 	def __init__(self, config, config_unet):
# 		super().__init__()
#
# 		self.config_unet = config_unet
# 		self.config = config
# 		if config['Regularized_class']:
# 			self.class_embedding = RegularizedEmbedding(config['n_classes'], config['class_dim'], config['class_std'])
# 		else:
# 			self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
#
# 		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
# 		self.generator_antomy = UNet(self.config_unet)
# 		self.segmentor = Segmentntor(self.config_unet)
# 		self.generator = Generator(config['n_adain_layers'], config['adain_dim'], config['img_shape'][2], self.config_unet['out_channels'])
# 		# self.round_act = nn.ReLU()
# 		self.rounding = Rounding()
# 		self.flatten = nn.Flatten()
#
# 	def forward(self, img, class_id, img_u, class_id_u):
# 		# print(img.size(), "img size,")
# 		# print(class_id.size(), "size id labaled")
# 		# print(class_id_u.size(), "class_id_u size")
# 		# print(torch.count_nonzero(class_id), "class id none zero")
# 		# print(torch.count_nonzero(class_id_u), "class id u none zero")
# 		# print(img_u.size(), "img_u.size()")
# 		anatomy_img = self.generator_antomy(img)
# 		class_code = self.class_embedding(class_id)
#
# 		anatomy_img_u = self.generator_antomy(img_u)
# 		class_code_u = self.class_embedding(class_id_u)
#
# 		# print(class_id, "class_id,")
# 		if self.config['round']:
# 			# print(anatomy_img[anatomy_img > 0.5], "before")
# 			idx = anatomy_img > 0.5
# 			anatomy_img = self.rounding(anatomy_img)
# 			anatomy_img_u = self.rounding(anatomy_img_u)
#
# 		# get the full image
# 		content_code = self.flatten(anatomy_img)
# 		content_code_u = self.flatten(anatomy_img_u)
#
# 		class_adain_params = self.modulation(class_code)
# 		class_adain_params_u = self.modulation(class_code_u)
#
# 		generated_img = self.generator(anatomy_img, class_adain_params)
# 		generated_img_u = self.generator(anatomy_img_u, class_adain_params_u)
#
# 		if self.config['segmentor_gard']:
# 			segmentor_input = anatomy_img
# 			segmentor_input_u = anatomy_img_u
# 			# print("here")
# 		else:
# 			segmentor_input = anatomy_img.detach()
# 			segmentor_input_u = anatomy_img_u.detach()
#
#
#
# 		mask = self.segmentor(segmentor_input)
# 		mask_u = self.segmentor(segmentor_input_u)
#
# 		return {
# 			'img': generated_img,
# 			'img_u': generated_img_u,
# 			'anatomy_img': anatomy_img,
# 			'anatomy_img_u': anatomy_img_u,
# 			'mask': mask,
# 			'mask_u': mask_u,
# 			'class_code': class_code,
# 			'class_code_u': class_code_u,
# 			'content_code': content_code,
# 			'content_code_u': content_code_u
# 		}
#
# 	def init(self):
# 		self.apply(self.weights_init)
#
# 	@staticmethod
# 	def weights_init(m):
# 		if isinstance(m, nn.Embedding):
# 			nn.init.uniform_(m.weight, a=-0.05, b=0.05)
# 			# nn.init.uniform_(m.weight, a = 0.45, b = 0.55)
#
#
#
#
# class ULordModel3D(nn.Module):
#
# 	def __init__(self, config, config_unet):
# 		super().__init__()
#
# 		self.config_unet = config_unet
# 		self.config = config
# 		if config['Regularized_class']:
# 			self.class_embedding = RegularizedEmbedding(config['n_classes'], config['class_dim'], config['class_std'])
# 		else:
# 			self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
#
# 		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
# 		self.generator_antomy = UNet(self.config_unet)
# 		self.segmentor = Segmentntor(self.config_unet)
# 		# self.generator = Generator(config['n_adain_layers'], config['adain_dim'], 1, self.config_unet['out_channels'])
# 		self.generator = Generator3D(config['n_adain_layers'], config['adain_dim'], 1, self.config_unet['out_channels'])
# 		# self.round_act = nn.ReLU()
# 		self.rounding = Rounding()
# 		self.softmax = nn.Softmax(dim = 1)
# 		self.flatten = nn.Flatten()
#
# 	def forward(self, img, class_id, img_u, class_id_u):
# 		depth = img.size()[2]
# 		class_id = class_id.repeat_interleave(depth)
# 		class_id_u = class_id_u.repeat_interleave(depth)
# 		anatomy_img = self.generator_antomy(img)
# 		class_code = self.class_embedding(class_id)
#
# 		s_a_img = anatomy_img.size()
#
# 		# anatomy_img = torch.reshape(anatomy_img, (s_a_img[0] * self.config_unet['in_channels'], self.config_unet['anatomy_rep'], s_a_img[2], s_a_img[3]))
# 		# anatomy_img = self.softmax(anatomy_img)
# 		# anatomy_img = torch.reshape(anatomy_img, (s_a_img[0], s_a_img[1], s_a_img[2], s_a_img[3]))
#
# 		anatomy_img_u = self.generator_antomy(img_u)
# 		class_code_u = self.class_embedding(class_id_u)
# 		s_a_img_u = anatomy_img_u.size()
#
# 		# apply softmax on the z axis
# 		# anatomy_img_u = torch.reshape(anatomy_img_u, (s_a_img_u[0] * self.config_unet['in_channels'], self.config_unet['anatomy_rep'], s_a_img_u[2], s_a_img_u[3]))
# 		# anatomy_img_u = self.softmax(anatomy_img_u)
# 		# anatomy_img_u = torch.reshape(anatomy_img_u, (s_a_img_u[0], s_a_img_u[1], s_a_img_u[2], s_a_img_u[3]))
# 		#
# 		# anatomy_img = torch.reshape(anatomy_img, (
# 		# s_a_img[0] * self.config_unet['in_channels'], self.config_unet['anatomy_rep'], s_a_img[2], s_a_img[3]))
# 		# anatomy_img = self.softmax(anatomy_img)
# 		# anatomy_img = torch.reshape(anatomy_img, (s_a_img[0], s_a_img[1], s_a_img[2], s_a_img[3]))
#
# 		# apply rounding
# 		if self.config['round']:
#
# 			anatomy_img = self.rounding(anatomy_img)
# 			anatomy_img_u = self.rounding(anatomy_img_u)
#
# 		# get the full image
# 		content_code = self.flatten(anatomy_img)
# 		content_code_u = self.flatten(anatomy_img_u)
#
# 		class_adain_params = self.modulation(class_code)
# 		class_adain_params_u = self.modulation(class_code_u)
#
# 		# inp_gen = anatomy_img.contiguous().view(s_a_img[0] * s_a_img[2],s_a_img[1], s_a_img[3],s_a_img[4])
# 		# inp_gen_u = anatomy_img_u.contiguous().view(s_a_img_u[0] * s_a_img_u[2],s_a_img_u[1], s_a_img_u[3],s_a_img_u[4])
# 		# #
# 		inp_gen = anatomy_img
# 		inp_gen_u = anatomy_img_u
#
# 		generated_img = self.generator(inp_gen, class_adain_params)
# 		generated_img_u = self.generator(inp_gen_u, class_adain_params_u)
#
#
# 		# print(generated_img_u.size(), " generated_img usizeeeee anat")
#
# 		# generated_img = generated_img.contiguous().view(s_a_img[0], 1, s_a_img[2],s_a_img[3], s_a_img[4])
# 		# generated_img_u = generated_img_u.contiguous().view(s_a_img_u[0] , 1, s_a_img_u[2], s_a_img_u[3], s_a_img_u[4])
#
# 		if self.config['segmentor_gard']:
# 			segmentor_input = anatomy_img
# 			segmentor_input_u = anatomy_img_u
# 			# print("here")
# 		else:
# 			segmentor_input = anatomy_img.detach()
# 			segmentor_input_u = anatomy_img_u.detach()
#
#
#
# 		mask = self.segmentor(segmentor_input)
# 		mask_u = self.segmentor(segmentor_input_u)
#
# 		return {
# 			'img': generated_img,
# 			'img_u': generated_img_u,
# 			'anatomy_img': anatomy_img,
# 			'anatomy_img_u': anatomy_img_u,
# 			'mask': mask,
# 			'mask_u': mask_u,
# 			'class_code': class_code,
# 			'class_code_u': class_code_u,
# 			'content_code': content_code,
# 			'content_code_u': content_code_u
# 		}
#
# 	def init(self):
# 		self.apply(self.weights_init)
#
# 	@staticmethod
# 	def weights_init(m):
# 		if isinstance(m, nn.Embedding):
# 			nn.init.uniform_(m.weight, a=-0.05, b=0.05)
# 			# nn.init.uniform_(m.weight, a = 0.45, b = 0.55)
#
#
#
# class UnetLatentModel(nn.Module):
#
# 	def __init__(self, config, config_unet):
# 		super().__init__()
#
# 		self.config_unet = config_unet
# 		self.config = config
# 		if config['Regularized_class']:
# 			self.class_embedding = RegularizedEmbedding(config['n_classes'], config['class_dim'], config['class_std'])
# 		else:
# 			self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
#
# 		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
# 		self.generator_antomy = UNet(self.config_unet)
# 		self.segmentor = Segmentntor(self.config_unet)
# 		self.generator = Generator(config['n_adain_layers'], config['adain_dim'], config['img_shape'], self.config_unet['out_channels'])
# 		# self.round_act = nn.ReLU()
# 		self.rounding = Rounding()
# 		self.flatten = nn.Flatten()
#
# 	def forward(self, img, class_id):
# 		# print(img.size(), "img size,")
# 		print(class_id, "class_id,")
# 		anatomy_img = self.generator_antomy(img)
# 		class_code = self.class_embedding(class_id) * 2
# 		print(class_code, "class_id,")
# 		# print(class_id, "class_id,")
# 		if self.config['round']:
# 			# print(anatomy_img[anatomy_img > 0.5], "before")
# 			idx = anatomy_img > 0.5
# 			anatomy_img = self.rounding(anatomy_img)
# 			# epsilon = torch.ones_like(anatomy_img)
# 			# epsilon = epsilon / 10000
# 			# one_matrix = one_matrix / 2
# 			# anatomy_img_subtracted =
# 			# anatomy_img_subtracted_r = self.round_act(anatomy_img_subtracted)
# 			# m = anatomy_img_subtracted + epsilon
# 			# anatomy_img = torch.divide(anatomy_img_subtracted_r, m)
# 			# print(anatomy_img[idx], "after")
# 			# print(anatomy_img[0, :, 0, 0], "after_fround")
#
# 		# get the full image
# 		content_code = self.flatten(anatomy_img)
# 		class_adain_params = self.modulation(class_code)
# 		# print(class_adain_params,"class adain paramters" )
# 		# print(class_code,"class_code," )
# 		generated_img = self.generator(anatomy_img, class_adain_params)
# 		mask = self.segmentor(anatomy_img)
# 		return {
# 			'img': generated_img,
# 			'anatomy_img': anatomy_img,
# 			'mask': mask,
# 			'class_code': class_code,
# 			'content_code': content_code
# 		}
#
# 	def init(self):
# 		self.apply(self.weights_init)
#
# 	@staticmethod
# 	def weights_init(m):
# 		if isinstance(m, nn.Embedding):
# 			# nn.init.uniform_(m.weight, a=-0.05, b=0.05)
# 			nn.init.uniform_(m.weight, a = 0.45, b = 0.55)
#
#
#
#
# class RegularizedEmbedding(nn.Module):
#
# 	def __init__(self, num_embeddings, embedding_dim, stddev):
# 		super().__init__()
#
# 		self.embedding = nn.Embedding(num_embeddings, embedding_dim)
# 		self.stddev = stddev
#
# 	def forward(self, x):
# 		x = self.embedding(x)
#
# 		if self.training and self.stddev != 0:
# 			noise = torch.zeros_like(x)
# 			noise.normal_(mean = 0, std = self.stddev)
#
# 			x = x + noise
#
# 		return x
#
#
# class Rounding(nn.Module):
#
# 	def __init__(self):
# 		super().__init__()
#
# 	def forward(self, x):
#
# 		Rounding_factor = torch.zeros_like(x, requires_grad= False)
# 		tensor = torch.tensor(x, requires_grad = False)
# 		# print(tensor.size(), "tensor")
# 		Rounding_factor[tensor <= 0.5] = tensor[tensor <= 0.5] * (-1)
# 		Rounding_factor[tensor > 0.5] = 1 - tensor[tensor > 0.5]
#
# 		x = x + Rounding_factor
#
# 		return x
#
#
#
# class Modulation(nn.Module):
#
# 	def __init__(self, code_dim, n_adain_layers, adain_dim):
# 		super().__init__()
#
# 		self.__n_adain_layers = n_adain_layers
# 		self.__adain_dim = adain_dim
#
# 		self.adain_per_layer = nn.ModuleList([
# 			nn.Linear(in_features = code_dim, out_features = adain_dim * 2)
# 			for _ in range(n_adain_layers)
# 		])
#
# 	def forward(self, x):
#
# 		adain_all = torch.cat([f(x) for f in self.adain_per_layer], dim = -1)
# 		adain_params = adain_all.reshape(-1, self.__n_adain_layers, self.__adain_dim, 2)
#
# 		return adain_params
#
#
# class Generator(nn.Module):
#
# 	def __init__(self, n_adain_layers, adain_dim, out_channel, in_channel):
# 		super().__init__()
#
# 		self.__adain_dim = adain_dim
#
# 		self.adain_conv_layers = nn.ModuleList()
# 		for i in range(n_adain_layers):
# 			if i == 0:
# 				in_channels_adain = in_channel
# 			else:
# 				in_channels_adain = adain_dim
#
# 			self.adain_conv_layers += [
# 				nn.Conv2d(in_channels = in_channels_adain, out_channels=adain_dim, padding= 1, kernel_size=3),
# 				nn.LeakyReLU(),
# 				AdaptiveInstanceNorm2d(adain_layer_idx = i)
# 			]
#
# 		self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)
#
# 		self.last_conv_layers = nn.Sequential(
# 			nn.Conv2d(in_channels = adain_dim, out_channels = 64, padding = 2, kernel_size = 5),
# 			nn.LeakyReLU(),
#
# 			nn.Conv2d(in_channels = 64, out_channels = out_channel, padding = 3, kernel_size = 7),
# 			nn.Sigmoid()
# 		)
#
# 	def assign_adain_params(self, adain_params):
# 		for m in self.adain_conv_layers.modules():
# 			if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
# 				m.bias = adain_params[:, m.adain_layer_idx, :, 0]
# 				m.weight = adain_params[:, m.adain_layer_idx, :, 1]
#
# 	def forward(self, x, class_adain_params):
# 		self.assign_adain_params(class_adain_params)
# 		x = self.adain_conv_layers(x)
# 		x = self.last_conv_layers(x)
# 		return x
#
# class Generator3D(nn.Module):
#
# 	def __init__(self, n_adain_layers, adain_dim, out_channel, in_channel):
# 		super().__init__()
#
# 		self.__adain_dim = adain_dim
#
# 		self.adain_conv_layers = nn.ModuleList()
# 		for i in range(n_adain_layers):
# 			if i == 0:
# 				in_channels_adain = in_channel
# 			else:
# 				in_channels_adain = adain_dim
#
# 			self.adain_conv_layers += [
# 				nn.Conv3d(in_channels=in_channels_adain, out_channels=adain_dim, padding=1, kernel_size=3),
# 				nn.LeakyReLU(),
# 				AdaptiveInstanceNorm3d(adain_layer_idx=i)
# 			]
#
# 		self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)
#
# 		self.last_conv_layers = nn.Sequential(
# 			nn.Conv3d(in_channels=adain_dim, out_channels=64, padding=2, kernel_size=5),
# 			nn.LeakyReLU(),
#
# 			nn.Conv3d(in_channels=64, out_channels=out_channel, padding=3, kernel_size=7),
# 			nn.Sigmoid()
# 		)
#
# 	def assign_adain_params(self, adain_params):
# 		for m in self.adain_conv_layers.modules():
# 			if m.__class__.__name__ == "AdaptiveInstanceNorm3d":
# 				m.bias = adain_params[:, m.adain_layer_idx, :, 0]
# 				m.weight = adain_params[:, m.adain_layer_idx, :, 1]
#
# 	def forward(self, x, class_adain_params):
# 		self.assign_adain_params(class_adain_params)
# 		x = self.adain_conv_layers(x)
# 		x = self.last_conv_layers(x)
# 		return x
#
# class Encoder(nn.Module):
#
# 	def __init__(self, img_shape, code_dim):
# 		super().__init__()
#
# 		self.conv_layers = nn.Sequential(
# 			nn.Conv2d(in_channels=img_shape[-1], out_channels=64, kernel_size=7, stride=1, padding=3),
# 			nn.LeakyReLU(),
#
# 			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
# 			nn.LeakyReLU(),
#
# 			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
# 			nn.LeakyReLU(),
#
# 			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
# 			nn.LeakyReLU(),
#
# 			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
# 			nn.LeakyReLU()
# 		)
#
# 		self.fc_layers = nn.Sequential(
# 			nn.Linear(in_features=4096, out_features=256),
# 			nn.LeakyReLU(),
#
# 			nn.Linear(in_features=256, out_features=256),
# 			nn.LeakyReLU(),
#
# 			nn.Linear(256, code_dim)
# 		)
#
# 	def forward(self, x):
# 		batch_size = x.shape[0]
# 		x = self.conv_layers(x)
# 		x = x.contiguous().view((batch_size, -1))
#
# 		x = self.fc_layers(x)
# 		return x
#
# class AdaptiveInstanceNorm3d(nn.Module):
#
# 	def __init__(self, adain_layer_idx):
# 		super().__init__()
# 		self.weight = None
# 		self.bias = None
# 		self.adain_layer_idx = adain_layer_idx
#
# 	def forward(self, x):
# 		b, c, D = x.shape[0], x.shape[1], x.shape[2]
#
# 		x_reshaped = x.contiguous().view(1, b * c * D , *x.shape[3:])
# 		weight = self.weight.contiguous().view(-1)
# 		bias = self.bias.contiguous().view(-1)
#
# 		out = F.batch_norm(
# 			x_reshaped, running_mean=None, running_var=None,
# 			weight=weight, bias=bias, training=True
# 		)
#
# 		out = out.view(b, c, D,  *x.shape[3:])
# 		return out
#
# class AdaptiveInstanceNorm2d(nn.Module):
#
# 	def __init__(self, adain_layer_idx):
# 		super().__init__()
# 		self.weight = None
# 		self.bias = None
# 		self.adain_layer_idx = adain_layer_idx
#
# 	def forward(self, x):
# 		b, c = x.shape[0], x.shape[1]
#
# 		x_reshaped = x.contiguous().view(1, b * c, *x.shape[2:])
# 		weight = self.weight.contiguous().view(-1)
# 		bias = self.bias.contiguous().view(-1)
#
# 		out = F.batch_norm(
# 			x_reshaped, running_mean=None, running_var=None,
# 			weight=weight, bias=bias, training=True
# 		)
#
# 		out = out.view(b, c, *x.shape[2:])
# 		return out
#
# class LambdaLayer(nn.Module):
# 	def __init__(self, lambd):
# 		super(LambdaLayer, self).__init__()
# 		self.lambd = lambd
#
# 	def forward(self, x):
# 		return self.lambd(x)
#
# class NetVGGFeatures(nn.Module):
#
# 	def __init__(self, layer_ids):
# 		super().__init__()
#
# 		self.vggnet = models.vgg16(pretrained = True)
# 		self.layer_ids = layer_ids
#
# 	def forward(self, img):
# 		output = []
#
# 		if img.shape[1] == 1:
# 			x = LambdaLayer(lambda t: torch.repeat_interleave(t, 3, 1))(img)
# 		else:
# 			x = img
#
# 		for i in range(self.layer_ids[-1] + 1):
# 			x = self.vggnet.features[i](x)
#
# 			if i in self.layer_ids:
# 				output.append(x)
#
# 		return output
#
#
# class VGGDistance(nn.Module):
#
# 	def __init__(self, layer_ids):
# 		super().__init__()
#
# 		self.vgg = NetVGGFeatures(layer_ids)
# 		self.layer_ids = layer_ids
#
# 	def forward(self, I1, I2):
# 		b_sz = I1.size(0)
# 		f1 = self.vgg(I1)
# 		f2 = self.vgg(I2)
#
# 		loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)
#
# 		for i in range(len(self.layer_ids)):
# 			layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
# 			loss = loss + layer_loss
#
# 		return loss.mean()