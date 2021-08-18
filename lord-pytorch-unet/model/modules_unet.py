import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import pickle
from torch.autograd import Function
from model.unet2d_model import UNet, Segmentntor
import numpy as np
import time
import random
from model.utils_unet import MyRotationTransformCL3D, MyRotationTransformC

CUDA_LAUNCH_BLOCKING=1

def pickle_load(in_file):
	with open(in_file, "rb") as opened_file:
		return pickle.load(opened_file)

# class my_round_func(Function):
#
# 	@staticmethod
# 	def forward(ctx, input):
# 		ctx.input = input
# 		return torch.round(input)
#
# 	@staticmethod
# 	def backward(ctx, grad_output):
# 		grad_input = grad_output.clone()
# 		return grad_input


class UNetS(nn.Module):

	def __init__(self, config_unet, dim):
		super().__init__()
		self.dim = dim
		self.config_unet = config_unet
		self.unet = UNet(self.config_unet, self.dim)
		self.segmentor = Segmentntor(self.config_unet, self.dim)

	def forward(self, img):
		x = self.unet(img)
		x = self.segmentor(x)

		return {
			'mask': x}

	def init(self):
		self.apply(self.weights_init)

	@staticmethod
	def weights_init(m):
		if isinstance(m, nn.Embedding):
			nn.init.uniform_(m.weight, a=-0.05, b=0.05)

class UNetSC(nn.Module):

	def __init__(self, config_unet, dim):
		super().__init__()
		self.dim = dim
		self.config_unet = config_unet
		self.unet = UNet(self.config_unet, self.dim)
		self.segmentor = Segmentntor(self.config_unet, self.dim)
		if self.dim == 3:
			self.rotation_transform = MyRotationTransformCL3D()
		else:
			self.rotation_transform = MyRotationTransformC()

		self.angels = [-90, 0, 90]

	def forward(self, img,classes1, img_u, classes2):

		if self.training:
			angle1 = random.choice(self.angels)
			angle2 = random.choice(self.angels)
			img_u_1 = self.rotation_transform(img_u, angle1)
			img_u_2 = self.rotation_transform(img_u, angle2)
		else:
			img_u_1 = img_u
			img_u_2 = img_u
			angle1 = 0
			angle2 = 0



		x_u_1 = self.unet(img_u_1)
		x_u_2 = self.unet(img_u_2)
		x_u_1 = self.segmentor(x_u_1)
		x_u_2 = self.segmentor(x_u_2)

		x = self.unet(img)
		x = self.segmentor(x)

		return {
			'mask': x,
			'mask_u_1': x_u_1,
			'mask_u_2': x_u_2,
			'angle1': angle1,
			'angle2': angle2,
		}

	def init(self):
		self.apply(self.weights_init)

	@staticmethod
	def weights_init(m):
		if isinstance(m, nn.Embedding):
			nn.init.uniform_(m.weight, a=-0.05, b=0.05)


class UCLordModel(nn.Module):

	def __init__(self, config, config_unet, dim):
		super().__init__()

		self.dim = dim
		self.config_unet = config_unet
		self.config = config
		self.angles = [-90, 0, 90, 180]
		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
		if self.dim == 3:
			self.generator_antomy = UNet(self.config_unet, 3)
			self.segmentor = Segmentntor(self.config_unet, 3)
			self.generator = Generator3D(config['n_adain_layers'],
										 config['adain_dim'], 1,
										 self.config_unet['out_channels'],
										 False)
			self.rotation_transform = MyRotationTransformCL3D()
		else:
			self.generator_antomy = UNet(self.config_unet, 2)
			self.segmentor = Segmentntor(self.config_unet, 2)
			self.generator = Generator(config['n_adain_layers'],
									   config['adain_dim'],
									   config['img_shape'][2],
									   self.config_unet['out_channels'])
			self.rotation_transform = MyRotationTransformC()

		self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
		self.rounding = Rounding()
		self.flatten = nn.Flatten()



	def forward(self, img, class_id, img_u, class_id_u):
		depth = img.size()[2]
		if self.dim == 3:
			class_id = class_id.repeat_interleave(depth)
			class_id_u = class_id_u.repeat_interleave(depth)

		if self.training:
			angle1 = random.choice(self.angles)
			angle2 = random.choice(self.angles)
			img_u_1 = self.rotation_transform(img_u, angle1)
			img_u_2 = self.rotation_transform(img_u, angle2)
		else:
			img_u_1 =  img_u
			img_u_2 = img_u
			angle1 = 0
			angle2 = 0

		anatomy_img = self.generator_antomy(img)
		class_code = self.class_embedding(class_id)



		anatomy_img_u_1 = self.generator_antomy(img_u_1)
		anatomy_img_u_2 = self.generator_antomy(img_u_2)
		class_code_u = self.class_embedding(class_id_u)

		# apply rounding
		if self.config['round']:
			anatomy_img = self.rounding(anatomy_img)
			anatomy_img_u_1 = self.rounding(anatomy_img_u_1)
			anatomy_img_u_2 = self.rounding(anatomy_img_u_2)

		# get the full image
		content_code = self.flatten(anatomy_img)
		content_code_u = self.flatten(anatomy_img_u_1)

		class_adain_params = self.modulation(class_code)
		class_adain_params_u = self.modulation(class_code_u)

		inp_gen = anatomy_img
		inp_gen_u = anatomy_img_u_1

		generated_img = self.generator(inp_gen, class_adain_params)
		generated_img_u = self.generator(inp_gen_u, class_adain_params_u)


		if self.config['segmentor_gard']:
			segmentor_input = anatomy_img
			segmentor_input_u_1 = anatomy_img_u_1
			segmentor_input_u_2 = anatomy_img_u_2
		else:
			segmentor_input = anatomy_img.detach()
			segmentor_input_u_1 = anatomy_img_u_1.detach()
			segmentor_input_u_2 = anatomy_img_u_2.detach()



		mask = self.segmentor(segmentor_input)
		mask_u_1 = self.segmentor(segmentor_input_u_1)
		mask_u_2 = self.segmentor(segmentor_input_u_2)

		return {
			'img': generated_img,
			'img_u': generated_img_u,
			'anatomy_img': anatomy_img,
			'anatomy_img_u_1': anatomy_img_u_1,
			'anatomy_img_u_2': anatomy_img_u_2,
			'angle1': angle1,
			'angle2': angle2,
			'mask': mask,
			'mask_u_1': mask_u_1,
			'mask_u_2': mask_u_2,
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


class ULordModel(nn.Module):

	def __init__(self, config, config_unet, dim):
		super().__init__()

		self.config_unet = config_unet
		self.config = config
		self.dim = dim
		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])

		if self.dim == 3:
			self.generator_antomy = UNet(self.config_unet, 3)
			self.segmentor = Segmentntor(self.config_unet, 3)
			self.generator = Generator3D(config['n_adain_layers'],
										 config['adain_dim'], 1,
										 self.config_unet['out_channels'],
										 False)
		else:
			self.generator_antomy = UNet(self.config_unet, 2)
			self.segmentor = Segmentntor(self.config_unet, 2)
			self.generator = Generator(config['n_adain_layers'],
									   config['adain_dim'],
									   config['img_shape'][2],
									   self.config_unet['out_channels'])

		self.rounding = Rounding()
		self.flatten = nn.Flatten()
		print(config['n_classes'], config['class_dim'], "config['n_classes'], config['class_dim']")
		self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])

	def forward(self, img, class_id, img_u, class_id_u):
		depth = img.size()[2]
		if self.dim == 3:
			class_id = class_id.repeat_interleave(depth)
			class_id_u = class_id_u.repeat_interleave(depth)


		anatomy_img = self.generator_antomy(img)
		class_code = self.class_embedding(class_id)

		anatomy_img_u = self.generator_antomy(img_u)
		class_code_u = self.class_embedding(class_id_u)

		# apply rounding
		if self.config['round']:
			anatomy_img = self.rounding(anatomy_img)
			anatomy_img_u = self.rounding(anatomy_img_u)

		# get the full image
		content_code = self.flatten(anatomy_img)
		content_code_u = self.flatten(anatomy_img_u)

		class_adain_params = self.modulation(class_code)
		class_adain_params_u = self.modulation(class_code_u)

		if self.config['segmentor_gard']:
			inp_gen = anatomy_img
			inp_gen_u = anatomy_img_u
		else:
			inp_gen  = anatomy_img.detach()
			inp_gen_u  = anatomy_img_u.detach()


		generated_img = self.generator(inp_gen, class_adain_params)
		generated_img_u = self.generator(inp_gen_u, class_adain_params_u)

		segmentor_input = anatomy_img
		segmentor_input_u = anatomy_img_u

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
			nn.Conv3d(in_channels=adain_dim, out_channels = 64, padding=2, kernel_size=5),
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

	def forward(self, I1, I2, mask = None):

		if mask == None:
			b_sz = I1.size(0)
			f1 = self.vgg(I1)
			f2 = self.vgg(I2)

			# loss = torch.abs(I1 - I2).view(b_sz, -1)
			loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)
			print(loss.size(), "losssss")
			for i in range(len(self.layer_ids)):
				# layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1)
				layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
				loss = loss + layer_loss
				# print(loss.size(), "losssss")
			return loss.mean()
		else:
			# print(mask.min(), mask.max(), "min max before 1")
			#
			mask = mask.detach()
			mask_opp = 1 - mask
			mask_opp = mask_opp.detach()
			b_sz = I1.size(0)
			sum_mask = mask.view(b_sz, -1).sum(1) + 1
			sum_mask_opp = mask_opp.view(b_sz, -1).sum(1) + 1
			f1 = self.vgg(I1)
			f2 = self.vgg(I2)
			# m_f = f1
			# mo_f =f2
			m_f = self.vgg(mask)
			mo_f = self.vgg(mask_opp)

			# print(sum_mask.size(), "sum mask size 1")

			# loss = torch.abs(I1 - I2).view(b_sz, -1)
			l = torch.abs((I1 * mask) - (I2 * mask)).view(b_sz, -1).sum(1)
			# print(l.size(), "losssss")
			loss = torch.divide(l, sum_mask)
			# print(loss.size(), "losssss")
			loss = loss + torch.divide(torch.abs((I1 * mask_opp) - (I2 * mask_opp)).view(b_sz, -1).sum(1), sum_mask_opp)
			for i in range(len(self.layer_ids)):
				# layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1)
				# print("m_f check \n")
				# print(m_f[i].min(), m_f[i].max(), "min max before")
				m_f[i] = (m_f[i] - m_f[i].min()) / (m_f[i].max() - m_f[i].min())
				# print(m_f[i].min(),m_f[i].max(), "min max after")

				# print("mo_f check \n")
				# print(mo_f[i].min(), mo_f[i].max(), "min max before")
				mo_f[i] = (mo_f[i] - mo_f[i].min()) / (mo_f[i].max() - mo_f[i].min())
				# print(mo_f[i].min(), mo_f[i].max(), "min max after\n")

				mask_opp = torch.round(mo_f[i])
				mask = torch.round(m_f[i])

				sum_m_f = mask.view(b_sz, -1).sum(1) + 1
				sum_mo_f = mask_opp.view(b_sz, -1).sum(1) + 1

				# print(sum_m_f, "sum_m_f size")
				# print(sum_mo_f, "sum_m_f size")

				layer_loss = torch.divide(torch.abs((f1[i] * mask) - (f2[i] * mask)).view(b_sz, -1).sum(1), sum_m_f)
				# print(layer_loss, "layer loss")
				layer_loss = layer_loss + torch.divide(torch.abs((f1[i] * mask_opp) - (f2[i] * mask_opp)).view(b_sz, -1).sum(1), sum_mo_f)
				# print(layer_loss, "layer loss 2")
				loss = loss + layer_loss

			return loss.mean()
