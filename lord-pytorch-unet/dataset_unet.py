import os
import re
from abc import ABC, abstractmethod

import numpy as np
import imageio

import dlib
import h5py
import torchvision.datasets as dset
import pickle
import matplotlib.pyplot as plt
root = './data'
from skimage.feature import canny
from skimage.morphology import skeletonize

S_DATA = 10
SM_DATA = 100
M_DATA = 500
L_DATA = 7000
PART = True

supported_datasets = [
	'mnist',
	'smallnorb',
	'cars3d',
	'shapes3d',
	'celeba',
	'kth',
	'rafd',
	'embryos_dataset',
	'brain_dataset'
]

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)

def get_dataset(dataset_id, path=None):

	if dataset_id == 'embryos_dataset':
		return embryos_dataset(path)

	if dataset_id == 'brain_dataset':
		return brain_dataset(path)

	if dataset_id == 'mnist':
		return Mnist()

	if dataset_id == 'smallnorb':
		return SmallNorb(path)

	if dataset_id == 'cars3d':
		return Cars3D(path)

	if dataset_id == 'shapes3d':
		return Shapes3D(path)

	if dataset_id == 'celeba':
		return CelebA(path)

	if dataset_id == 'kth':
		return KTH(path)

	if dataset_id == 'rafd':
		return RaFD(path)

	raise Exception('unsupported dataset: %s' % dataset_id)



class DataSet(ABC):

	def __init__(self, base_dir=None):
		super().__init__()
		self._base_dir = base_dir

	@abstractmethod
	def read_images(self):
		pass

class brain_dataset(DataSet):
	"""
	take all what you get conctante it add class and deliver
	"""

	def __init__(self, base_dir):
		super().__init__(base_dir)

	def load_data(self, code_word, num_id, show_images):
		print("loading...")
		with open(os.path.join(self._base_dir, code_word, 'patches_dict.h5'), "rb") as opened_file:
			dataset = pickle.load(opened_file)

		with open(os.path.join(self._base_dir, code_word, 'patches_dict_gt.h5'), "rb") as opened_file:
			dataset_gt = pickle.load(opened_file)


		imgs = np.array(list(dataset.values()))
		imgs_gt = np.array(list(dataset_gt.values()))
		class_id = np.zeros((imgs.shape[0]))
		class_id = class_id + num_id

		print(f"imgs_{code_word}", np.unique(imgs), f"imgs_len_{code_word}:", imgs.shape[0])
		num_f1 = 0
		r = 3
		if show_images and imgs.shape[0] > (num_f1 + r):
			for i in range(r):
				if len(imgs[0].shape) > 2:
					plt.title(f"imgs_{code_word}_{num_f1 + i}")
					plt.imshow(imgs[num_f1 + i][:, :, 0], cmap='gray')
					plt.show()
				else:
					plt.title(f"imgs_{code_word}_{num_f1 + i}")
					plt.imshow(imgs[num_f1 + i], cmap='gray')
					plt.show()
		print("finish loading...")
		return imgs, imgs_gt, class_id

	def concat(self, img1, img2, img3, img1_gt, img2_gt, img3_gt, class1, class2, class3 ):
		print("start checking if the images are need to concat ...")
		if img1.shape[0] > 0 and img2.shape[0] > 0 and img3.shape[0] > 0:
			class_id = np.concatenate((class1, class2, class3), axis=0)
			imgs = np.concatenate((img1, img2, img3), axis=0)
			segs = np.concatenate((img1_gt, img2_gt,img3_gt ), axis=0)
			imgs = np.expand_dims(imgs, axis=-1)
			segs = np.expand_dims(segs, axis=-1)
			return imgs, segs, class_id, True
		elif img1.shape[0] > 0 and img2.shape[0] > 0:
			class_id = np.concatenate((class1, class2), axis=0)
			imgs = np.concatenate((img1, img2), axis=0)
			segs = np.concatenate((img1_gt, img2_gt), axis=0)
			imgs = np.expand_dims(imgs, axis=-1)
			segs = np.expand_dims(segs, axis=-1)
			return imgs, segs, class_id, True
		elif img1.shape[0] > 0:
			class_id = class1
			imgs = np.expand_dims(img1, axis=-1)
			segs = np.expand_dims(img1_gt, axis=-1)
			return imgs, segs, class_id, True
		else:
			return None, None, None, False


	def read_images(self):
		show_images = False
		print(self._base_dir, "base dir chech ")

		imgs_fr, imgs_fr_gt, class_fr_id  = self.load_data('FR', 0, show_images)
		imgs_trufi, imgs_trufi_gt, class_trufi_id  = self.load_data('TRUFI', 1, show_images)
		imgs_haste, imgs_haste_gt, class_haste_id  = self.load_data('HASTE', 2, show_images)
		print("start concat...")
		s = False
		if not (imgs_fr.shape[0] > 0 and imgs_haste.shape[0] > 0 and imgs_trufi.shape[0] == 0):
			imgs, segs, class_id, s = self.concat(imgs_fr,imgs_trufi,imgs_haste,imgs_fr_gt,imgs_trufi_gt,imgs_haste_gt,class_fr_id,class_trufi_id,class_haste_id)
			print(f"s{s}")
		if not s:
			imgs, segs, class_id, s = self.concat(imgs_fr,imgs_haste, imgs_trufi,  imgs_fr_gt,imgs_haste_gt, imgs_trufi_gt, class_fr_id,class_haste_id, class_trufi_id)
		if not s:
			imgs, segs, class_id, s = self.concat(imgs_haste, imgs_trufi,imgs_fr, imgs_haste_gt, imgs_trufi_gt, imgs_fr_gt, class_haste_id, class_trufi_id, class_fr_id)
		if not s:
			imgs, segs, class_id, s = self.concat( imgs_trufi, imgs_fr,imgs_haste, imgs_trufi_gt, imgs_fr_gt, imgs_haste_gt, class_trufi_id, class_fr_id, class_haste_id)
		print("finish concat ...")
		print(f"shape imgs:{imgs.shape}", f"shape segs:{segs.shape}", f"shape classs id:{class_id.shape}")
		return imgs,class_id, segs

# class embryos_dataset(DataSet):
#
# 	def __init__(self, base_dir):
# 		super().__init__(base_dir)
# 		#TODO change size data
#
#
# 	def read_images(self, size_jump = 1 , dim3d = False):
# 		# print(os.path.join(self._base_dir,'plcenta','fetal_data.h5'), "aaaa")
#
# 		# load data set
# 		dim3d = True
# 		show_images = False
# 		# load data set
# 		with open(os.path.join(self._base_dir, 'placenta', 'fetal_data_patches.h5'), "rb") as opened_file:
# 			fiesta_dataset = pickle.load(opened_file)
#
# 		with open(os.path.join(self._base_dir, 'TRUFI', 'fetal_data_patches.h5'), "rb") as opened_file:
# 			trufi_dataset = pickle.load(opened_file)
#
# 		with open(os.path.join(self._base_dir, 'placenta', 'fetal_data_patches_gt.h5'), "rb") as opened_file:
# 			fiesta_dataset_gt = pickle.load(opened_file)
#
# 		with open(os.path.join(self._base_dir, 'TRUFI', 'fetal_data_patches_gt.h5'), "rb") as opened_file:
# 			trufi_dataset_gt = pickle.load(opened_file)
#
#
#
# 		#  load fiest
# 		imgs_fiesta = np.array(list(fiesta_dataset.values()))
# 		imgs_fiesta_gt = np.array(list(fiesta_dataset_gt.values()))
# 		class_id_fiesta = np.zeros((imgs_fiesta.shape[0]))
# 		print(imgs_fiesta.shape, "imgs_fiesta.shape")
# 		print(imgs_fiesta_gt.shape, "imgs_fiesta_gt.shape")
# 		if show_images:
# 			num_f1 = 101
# 			for i in range(3):
# 				if dim3d:
# 					plt.title(f"imgs_fiesta_{num_f1 + i}")
# 					plt.imshow(imgs_fiesta[num_f1 + i][:,:,0], cmap='gray')
# 					plt.show()
# 				else:
# 					plt.title(f"imgs_fiesta_{num_f1 + i}")
# 					plt.imshow(imgs_fiesta[num_f1 + i], cmap='gray')
# 					plt.show()
#
# 		# load trufi
# 		imgs_trufi = np.array(list(trufi_dataset.values()))
#
# 		imgs_trufi_gt = np.array(list(trufi_dataset_gt.values()))
# 		class_id_trufi = np.ones((imgs_trufi.shape[0]))
# 		print(imgs_trufi.shape, "imgs_trufi.shape")
# 		print(imgs_trufi_gt.shape, "imgs_trufi_gt.shape")
# 		if show_images:
# 			num_t1 = 100
# 			for i in range(3):
# 				if dim3d:
# 					plt.title(f"imgs_trufi_{num_t1  + i}")
# 					plt.imshow(imgs_trufi[num_t1 + i][:, :, 0], cmap='gray')
# 					plt.show()
# 				else:
# 					plt.title(f"imgs_trufi_{num_t1  + i}")
# 					plt.imshow(imgs_trufi[num_t1  + i], cmap='gray')
# 					plt.show()
#
# 		# concatnate all
# 		class_id = np.concatenate((class_id_fiesta, class_id_trufi),axis=0)
# 		imgs = np.concatenate((imgs_fiesta, imgs_trufi),axis=0)
# 		segs = np.concatenate((imgs_fiesta_gt, imgs_trufi_gt),axis=0)
# 		imgs = np.expand_dims(imgs,axis=-1)
# 		segs = np.expand_dims(segs,axis=-1)
#
#
# 		# concatante part
# 		print(imgs_fiesta[: :size_jump].shape, "imgs_fiesta[: :self.size_jump]")
# 		print(imgs_trufi[: :size_jump].shape, "imgs_trufi[: :self.size_jump]")
# 		print(imgs_fiesta_gt[::size_jump].shape, "imgs_fiesta_gt[::size_jump]")
# 		print(imgs_trufi_gt[::size_jump].shape, "imgs_trufi_gt[: :self.size_jump]")
#
# 		imgs_fiesta_part = imgs_fiesta[::size_jump]
# 		class_id_fiesta_part = class_id_fiesta[::size_jump]
# 		imgs_fiesta_gt_part = imgs_fiesta_gt[::size_jump]
#
# 		imgs_trufi_part = imgs_trufi[: :size_jump]
# 		class_id_trufi_part = class_id_trufi[::size_jump]
# 		imgs_trufi_gt_part = imgs_trufi_gt[::size_jump]
#
#
# 		min_size = np.min((imgs_trufi_part.shape[0], imgs_fiesta_part.shape[0]))
# 		print(min_size, "min size")
#
# 		imgs_part = np.concatenate((imgs_fiesta_part[: min_size],imgs_trufi_part[: min_size]), axis=0)
# 		segs_part = np.concatenate((imgs_fiesta_gt_part [: min_size],imgs_trufi_gt_part[: min_size]), axis=0)
# 		class_id_part = np.concatenate((class_id_fiesta_part[0: min_size],  class_id_trufi_part[0: min_size]),axis = 0)
#
# 		imgs_part = np.expand_dims(imgs_part ,axis=-1)
# 		segs_part = np.expand_dims(segs_part ,axis=-1)
#
# 		# normlaize data
# 		imgs_part = imgs_part - np.min(imgs_part)
# 		imgs_part = imgs_part / np.max(imgs_part)
#
# 		# if show_images:
# 		img_part =  0
# 		if show_images:
# 			for i in range(3):
# 				# plt.title(f"imgs_trufi_{img_part + i}")
# 				plt.imshow(imgs_part[img_part + i][:,:,0], cmap='gray')
# 				plt.show()
# 				plt.imshow(segs_part[img_part + i][:,:,0], cmap='gray')
#
# 				plt.show()
#
#
# 		imgs = imgs - np.min(imgs)
# 		imgs = imgs / np.max(imgs)
#
#
# 		PART = True
# 		if PART:
# 			return imgs_part, class_id_part, segs_part
#
# 		else:
# 			return imgs, class_id, segs

class embryos_dataset(DataSet):
	"""
	take all what you get conctante it add class and deliver
	"""

	def __init__(self, base_dir):
		super().__init__(base_dir)


	def read_images(self):
		# print(os.path.join(self._base_dir,'plcenta','fetal_data.h5'), "aaaa")
		show_images = False
		# load data set

		with open(os.path.join(self._base_dir, 'placenta', 'patches_dict.h5'), "rb") as opened_file:
			fiesta_dataset = pickle.load(opened_file)

		with open(os.path.join(self._base_dir, 'placenta', 'patches_dict_gt.h5'), "rb") as opened_file:
			fiesta_dataset_gt = pickle.load(opened_file)

		print(self._base_dir, "base dir chech ")
		imgs_fiesta = np.array(list(fiesta_dataset.values()))
		imgs_fiesta_gt = np.array(list(fiesta_dataset_gt.values()))
		# print(imgs_fiesta_gt , "img fiesta gt")
		class_id_fiesta = np.zeros((imgs_fiesta.shape[0]))
		print("imgs_fiesta unique:", imgs_fiesta.shape)
		if show_images and imgs_fiesta.shape[0] > 0:
			num_f1 = 0
			for i in range(3):
				plt.title(f"imgs_fiesta_{num_f1 + i}")
				plt.imshow(imgs_fiesta[num_f1 + i][:, :, 0], cmap='gray')
				plt.show()



		with open(os.path.join(self._base_dir, 'TRUFI', 'patches_dict.h5'), "rb") as opened_file:
			trufi_dataset = pickle.load(opened_file)

		with open(os.path.join(self._base_dir, 'TRUFI', 'patches_dict_gt.h5'), "rb") as opened_file:
			trufi_dataset_gt = pickle.load(opened_file)


		imgs_trufi = np.array(list(trufi_dataset.values()))
		imgs_trufi_gt = np.array(list(trufi_dataset_gt.values()))
		class_id_trufi = np.ones((imgs_trufi.shape[0]))
		print(f"shape trufi: {class_id_trufi.shape}")


		if show_images and imgs_trufi.shape[0] > 0:
			num_f1 = 10
			for i in range(3):
				plt.title(f"imgs_fiesta_{num_f1 + i}")
				plt.imshow(imgs_trufi[num_f1 + i][:, :, 0], cmap='gray')
				plt.show()

		if imgs_trufi.shape[0] > 0 and imgs_fiesta.shape[0] > 0:
			class_id = np.concatenate((class_id_fiesta, class_id_trufi), axis=0)
			imgs = np.concatenate((imgs_fiesta, imgs_trufi), axis=0)
			segs = np.concatenate((imgs_fiesta_gt, imgs_trufi_gt), axis=0)
			imgs = np.expand_dims(imgs, axis=-1)
			segs = np.expand_dims(segs, axis=-1)

			return imgs, class_id, segs

		elif imgs_trufi.shape[0] > 0 and imgs_fiesta.shape[0] == 0:
			imgs = np.expand_dims(imgs_trufi, axis=-1)
			segs = np.expand_dims(imgs_trufi_gt, axis=-1)
			return  imgs, class_id_trufi, segs

		elif imgs_trufi.shape[0] == 0 and imgs_fiesta.shape[0] > 0:
			imgs = np.expand_dims(imgs_fiesta , axis=-1)
			segs = np.expand_dims(imgs_fiesta_gt, axis=-1)

			return imgs, class_id_fiesta, segs




class Mnist(DataSet):

	def __init__(self):
		super().__init__()

	def read_images(self):

		d_train = dset.MNIST(root=root, train=True, download=True)
		d_test = dset.MNIST(root=root, train=False, download=True)

		x_train, y_train = d_train.train_data, d_train.train_labels
		x_test, y_test = d_test.test_data, d_test.test_labels


		print(d_train, d_test, "shapes Mnist")


		x = np.concatenate((x_train, x_test), axis=0)
		y = np.concatenate((y_train, y_test), axis=0)

		imgs = np.stack([cv2.resize(x[i], dsize=(64, 64)) for i in range(x.shape[0])], axis=0)
		imgs = np.expand_dims(imgs, axis=-1)

		classes = y
		contents = np.empty(shape=(x.shape[0], ), dtype=np.uint32)

		return imgs, classes, contents


class SmallNorb(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

	def __list_imgs(self):
		img_paths = []
		class_ids = []
		content_ids = []

		regex = re.compile('azimuth(\d+)_elevation(\d+)_lighting(\d+)_(\w+).jpg')
		for category in os.listdir(self._base_dir):
			for instance in os.listdir(os.path.join(self._base_dir, category)):
				for file_name in os.listdir(os.path.join(self._base_dir, category, instance)):
					img_path = os.path.join(self._base_dir, category, instance, file_name)
					azimuth, elevation, lighting, lt_rt = regex.match(file_name).groups()

					class_id = '_'.join((category, instance, elevation, lighting, lt_rt))
					content_id = azimuth

					img_paths.append(img_path)
					class_ids.append(class_id)
					content_ids.append(content_id)

		return img_paths, class_ids, content_ids

	def read_images(self):
		img_paths, class_ids, content_ids = self.__list_imgs()

		unique_class_ids = list(set(class_ids))
		unique_content_ids = list(set(content_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 1), dtype=np.uint8)
		classes = np.empty(shape=(len(img_paths), ), dtype=np.uint32)
		contents = np.empty(shape=(len(img_paths), ), dtype=np.uint32)

		for i in range(len(img_paths)):
			img = imageio.imread(img_paths[i])
			imgs[i, :, :, 0] = cv2.resize(img, dsize=(64, 64))

			classes[i] = unique_class_ids.index(class_ids[i])
			contents[i] = unique_content_ids.index(content_ids[i])

		return imgs, classes, contents


class Cars3D(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__data_path = os.path.join(base_dir, 'cars3d.npz')

	def read_images(self):
		imgs = np.load(self.__data_path)['imgs']
		classes = np.empty(shape=(imgs.shape[0], ), dtype=np.uint32)
		contents = np.empty(shape=(imgs.shape[0], ), dtype=np.uint32)

		for elevation in range(4):
			for azimuth in range(24):
				for object_id in range(183):
					img_idx = elevation * 24 * 183 + azimuth * 183 + object_id

					classes[img_idx] = object_id
					contents[img_idx] = elevation * 24 + azimuth

		return imgs, classes, contents


class Shapes3D(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__data_path = os.path.join(base_dir, '3dshapes.h5')

	def __img_index(self, floor_hue, wall_hue, object_hue, scale, shape, orientation):
		return (
			floor_hue * 10 * 10 * 8 * 4 * 15
			+ wall_hue * 10 * 8 * 4 * 15
			+ object_hue * 8 * 4 * 15
			+ scale * 4 * 15
			+ shape * 15
			+ orientation
		)

	def read_images(self):
		with h5py.File(self.__data_path, 'r') as data:
			imgs = data['images'][:]
			classes = np.empty(shape=(imgs.shape[0],), dtype=np.uint32)
			content_ids = dict()

			for floor_hue in range(10):
				for wall_hue in range(10):
					for object_hue in range(10):
						for scale in range(8):
							for shape in range(4):
								for orientation in range(15):
									img_idx = self.__img_index(floor_hue, wall_hue, object_hue, scale, shape, orientation)
									content_id = '_'.join((str(floor_hue), str(wall_hue), str(object_hue), str(scale), str(orientation)))

									classes[img_idx] = shape
									content_ids[img_idx] = content_id

			unique_content_ids = list(set(content_ids.values()))
			contents = np.empty(shape=(imgs.shape[0],), dtype=np.uint32)
			for img_idx, content_id in content_ids.items():
				contents[img_idx] = unique_content_ids.index(content_id)

			return imgs, classes, contents


class CelebA(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__imgs_dir = os.path.join(self._base_dir, 'Img', 'img_align_celeba_png.7z', 'img_align_celeba_png')
		self.__identity_map_path = os.path.join(self._base_dir, 'Anno', 'identity_CelebA.txt')

	def __list_imgs(self):
		with open(self.__identity_map_path, 'r') as fd:
			lines = fd.read().splitlines()

		img_paths = []
		class_ids = []

		for line in lines:
			img_name, class_id = line.split(' ')
			img_path = os.path.join(self.__imgs_dir, os.path.splitext(img_name)[0] + '.png')

			img_paths.append(img_path)
			class_ids.append(class_id)

		return img_paths, class_ids

	def read_images(self, crop_size=(128, 128), target_size=(64, 64)):
		img_paths, class_ids = self.__list_imgs()

		unique_class_ids = list(set(class_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 3), dtype=np.uint8)
		classes = np.empty(shape=(len(img_paths), ), dtype=np.uint32)
		contents = np.zeros(shape=(len(img_paths), ), dtype=np.uint32)

		for i in range(len(img_paths)):
			img = imageio.imread(img_paths[i])

			if crop_size:
				img = img[
					(img.shape[0] // 2 - crop_size[0] // 2):(img.shape[0] // 2 + crop_size[0] // 2),
					(img.shape[1] // 2 - crop_size[1] // 2):(img.shape[1] // 2 + crop_size[1] // 2)
				]

			if target_size:
				img = cv2.resize(img, dsize=target_size)

			imgs[i] = img
			classes[i] = unique_class_ids.index(class_ids[i])

		return imgs, classes, contents


class KTH(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__action_dir = os.path.join(self._base_dir, 'handwaving')
		self.__condition = 'd4'

	def __list_imgs(self):
		img_paths = []
		class_ids = []

		for class_id in os.listdir(self.__action_dir):
			for f in os.listdir(os.path.join(self.__action_dir, class_id, self.__condition)):
				img_paths.append(os.path.join(self.__action_dir, class_id, self.__condition, f))
				class_ids.append(class_id)

		return img_paths, class_ids

	def read_images(self):
		img_paths, class_ids = self.__list_imgs()

		unique_class_ids = list(set(class_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 1), dtype=np.uint8)
		classes = np.empty(shape=(len(img_paths), ), dtype=np.uint32)
		contents = np.zeros(shape=(len(img_paths), ), dtype=np.uint32)

		for i in range(len(img_paths)):
			imgs[i, :, :, 0] = cv2.cvtColor(cv2.imread(img_paths[i]), cv2.COLOR_BGR2GRAY)
			classes[i] = unique_class_ids.index(class_ids[i])

		return imgs, classes, contents


class RaFD(DataSet):
 
 
 
	def __init__(self, base_dir):
		super().__init__(base_dir)

	def __list_imgs(self):
		img_paths = []
		expression_ids = []

		regex = re.compile('Rafd(\d+)_(\d+)_(\w+)_(\w+)_(\w+)_(\w+).jpg')
		for file_name in os.listdir(self._base_dir):
			img_path = os.path.join(self._base_dir, file_name)
			idx, identity_id, description, gender, expression_id, angle = regex.match(file_name).groups()

			img_paths.append(img_path)
			expression_ids.append(expression_id)

		return img_paths, expression_ids

	def read_images(self):
		img_paths, expression_ids = self.__list_imgs()

		unique_expression_ids = list(set(expression_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 3), dtype=np.uint8)
		expressions = np.empty(shape=(len(img_paths), ), dtype=np.uint32)

		face_detector = dlib.get_frontal_face_detector()
		for i in range(len(img_paths)):
			img = imageio.imread(img_paths[i])

			detections, scores, weight_indices = face_detector.run(img, upsample_num_times=0, adjust_threshold=-1)
			face_bb = detections[np.argmax(scores)]

			top = max((face_bb.bottom() + face_bb.top()) // 2 - 681 // 2, 0)
			face = img[top:(top + 681), :]

			imgs[i] = cv2.resize(face, dsize=(64, 64))
			expressions[i] = unique_expression_ids.index(expression_ids[i])

		return imgs, expressions, np.zeros_like(expressions)

"""
!python lord_tf.py \
--base-dir='/content/drive/My Drive/Colab Notebooks' train \
--data-name='/content/drive/My Drive/Colab Notebooks/train_nir_mnist' \
--model-name='/content/drive/My Drive/Colab Notebooks/1_nir_mnist_model'
"""

"""
lord_tf.py --base-dir /cs/casmip/nirm/embryo_project/output preprocess
    --dataset-id smallnorb 
    --dataset-path /cs/casmip/nirm/embryo_project/SmallNorb/archive
    --data-name smallnorb
"""