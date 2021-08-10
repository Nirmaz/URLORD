import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
import random

class MyRotationTransform:
	"""Rotate by one of the given angles."""

	def __init__(self, angles):
		self.angles = angles

	def __call__(self, x):
		angle = random.choice(self.angles)
		# print(x['img'].size(),"size")

		x['img'] = TF.rotate(x['img'], angle)
		x['seg'] = TF.rotate(x['seg'], angle)

		return x


class MyRotationTransformCL3D:
	"""Rotate by one of the given angles."""


	def __call__(self, x, angle):
		# print((x['img'].dim()), "dimmmm"
		# rotation
		if x['img'].dim() == 5:
			# print(x['img'].size(),"size")
			x_tr = torch.squeeze(x['img'],dim=1)
			seg_tr = torch.squeeze(x['seg'],dim=1)
			# print(x_tr.size(), "x_tr before")
			# print(seg_tr.size(), "seg_tr before")
			x_tr = TF.rotate(x_tr, angle)
			seg_tr = TF.rotate(seg_tr, angle)
			x['img'] = torch.unsqueeze(x_tr, dim=1)
			x['seg'] = torch.unsqueeze(seg_tr, dim=1)
			# print(x['img'].size(), "x_tr after")
			# print(x['seg'].size(), "seg_tr")
		else:
			x['img'] = TF.rotate(x['img'], angle)
			x['seg'] = TF.rotate(x['seg'], angle)

		return x

class MyRotationTransform3D:
	"""Rotate by one of the given angles."""

	def __init__(self, angles):
		self.angles = angles

	def __call__(self, x):
		angle = random.choice(self.angles)
		# print((x['img'].dim()), "dimmmm")


		# rotation
		if x['img'].dim() == 5:
			# print(x['img'].size(),"size")
			x_tr = torch.squeeze(x['img'],dim=1)
			seg_tr = torch.squeeze(x['seg'],dim=1)
			# print(x_tr.size(), "x_tr before")
			# print(seg_tr.size(), "seg_tr before")
			x_tr = TF.rotate(x_tr, angle)
			seg_tr = TF.rotate(seg_tr, angle)
			x['img'] = torch.unsqueeze(x_tr, dim=1)
			x['seg'] = torch.unsqueeze(seg_tr, dim=1)
			# print(x['img'].size(), "x_tr after")
			# print(x['seg'].size(), "seg_tr")
		else:
			x['img'] = TF.rotate(x['img'], angle)
			x['seg'] = TF.rotate(x['seg'], angle)

		return x


class AverageMeter:

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class NamedTensorDataset(Dataset):

	def __init__(self, named_tensors, transform = None):
		assert all(list(named_tensors.values())[0].size(0) == tensor.size(0) for tensor in named_tensors.values())
		self.named_tensors = named_tensors
		self.transform = transform

	def __getitem__(self, index):
		# print({name: tensor[index] for name, tensor in self.named_tensors.items()})
		if self.transform == None:
			return {name: tensor[index] for name, tensor in self.named_tensors.items()}
		else:
			return self.transform({name: tensor[index] for name, tensor in self.named_tensors.items()})


	def __len__(self):
		return list(self.named_tensors.values())[0].size(0)

	def subset(self, indices):
		return NamedTensorDataset(self[indices])

# class ImbalancedDatasetSampler(Sampler):
#     """Samples elements randomly from a given list of indices for imbalanced dataset
#     Arguments:
#         indices: a list of indices
#         num_samples: number of samples to draw
#         callback_get_label: a callback-like function which takes two arguments - dataset and index
#     """
#
#     def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
#         # if indices is not provided, all elements in the dataset will be considered
#         self.indices = list(range(len(dataset))) if indices is None else indices
#
#         # define custom callback
#         self.callback_get_label = callback_get_label
#
#         # if num_samples is not provided, draw `len(indices)` samples in each iteration
#         self.num_samples = len(self.indices) if num_samples is None else num_samples
#
#         # distribution of classes in the dataset
#         df = pd.DataFrame()
#         df["label"] = self._get_labels(dataset)
#         df.index = self.indices
#         df = df.sort_index()
#
#         label_to_count = df["label"].value_counts()
#
#         weights = 1.0 / label_to_count[df["label"]]
#
#         self.weights = torch.DoubleTensor(weights.to_list())
#
#     def _get_labels(self, dataset):
#         if self.callback_get_label:
#             return self.callback_get_label(dataset)
#         elif isinstance(dataset, torchvision.datasets.MNIST):
#             return dataset.train_labels.tolist()
#         elif isinstance(dataset, torchvision.datasets.ImageFolder):
#             return [x[1] for x in dataset.imgs]
#         elif isinstance(dataset, torchvision.datasets.DatasetFolder):
#             return dataset.samples[:][1]
#         elif isinstance(dataset, torch.utils.data.Subset):
#             return dataset.dataset.imgs[:][1]
#         elif isinstance(dataset, torch.utils.data.Dataset):
#             return dataset.get_labels()
#         else:
#             raise NotImplementedError
#
#     def __iter__(self):
#         return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))
#
#     def __len__(self):
#         return self.num_samples
#

class ImbalancedDatasetSampler(Sampler):

	def __init__(self, dataset,classes ,num_samples: int = None, percent_list = None):
		# super().__init__(dataset)
		self.indices = list(range(len(dataset)))
		self.len_data = len(dataset)
		if num_samples == None:
			self.num_samples = self.len_data
		else:
			self.num_samples = num_samples

		if percent_list == None:
			values = np.zeros((self.len_data))
			values[:] = 1 / self.len_data
			weights = list(values)
		else:
			values = np.zeros((self.len_data))
			class_unique = np.unique(classes)
			for c in range(class_unique.shape[0]):
				sum_c = classes[classes == c].shape[0]
				values[classes == c] = percent_list[c] / sum_c

			weights = list(values)


		# print(self.num_samples, "num samples")
		# # classes of all the samples
		# self.classes = classes
		#
		# # distribution of classes in the dataset
		# df = pd.DataFrame()
		# df["label"] = self.classes
		# df.index = self.indices
		# print(df, "df before")
		# df = df.sort_index()
		# print(df, "df After")
		#
		# label_to_count = df["label"].value_counts()
		# print(label_to_count, "label_to_count")
		# weights = 1.0 / label_to_count[df["label"]]
		# print(weights, "weights")
		self.weights = torch.DoubleTensor(weights)


	def __iter__(self):
		a =  (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

		return a

	def __len__(self):

		return self.num_samples

