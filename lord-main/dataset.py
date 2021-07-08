import os
import re
from abc import ABC, abstractmethod
import pickle
import numpy as np
import imageio
import cv2
import dlib
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.datasets import mnist


supported_datasets = [
	'mnist',
	'smallnorb',
	'cars3d',
	'shapes3d',
	'celeba',
	'kth',
	'rafd',
	'embryo_dataset'
]


def get_dataset(dataset_id, path=None):
	print(dataset_id, "datasetid")
	if dataset_id == 'embryo_dataset':
		return embryos_dataset(path)

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



class embryos_dataset(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

	def read_images(self):
		# print(os.path.join(self._base_dir,'plcenta','fetal_data.h5'), "aaaa")

		trufi_index = 1
		fiesta_index = 0
		with open(os.path.join(self._base_dir,'placenta','fetal_data_patches.h5'), "rb") as opened_file:
			fiesta_dataset = pickle.load(opened_file)

		with open(os.path.join(self._base_dir,'TRUFI','fetal_data_patches.h5'), "rb") as opened_file:
			trufi_dataset = pickle.load(opened_file)

		# print(np.unique(trufi_dataset), " trufi dataset unique")
		imgs_fiesta = np.array(list(fiesta_dataset.values()))
		print(imgs_fiesta.shape, "img fiest shape")
		imgs_fiesta = imgs_fiesta.transpose((0, 3,1,2))
		data_size = imgs_fiesta.shape[0] * imgs_fiesta.shape[1]
		imgs_fiesta = imgs_fiesta.reshape((data_size, imgs_fiesta.shape[2],imgs_fiesta.shape[3]))
		print(imgs_fiesta.shape, "imgs fiesta shpe")
		class_id_fiesta = np.zeros((imgs_fiesta.shape[0]))
		print(class_id_fiesta.shape, "class fiesta shape")
		num_f1 = 3001
		plt.title(f"imgs_fiesta_{num_f1}")
		plt.imshow(imgs_fiesta[num_f1], cmap='gray')
		# plt.show()


		imgs_trufi = np.array(list(trufi_dataset.values()))
		imgs_trufi = imgs_trufi.transpose((0, 3,1,2))

		print(imgs_trufi.shape, "img trufi shape")
		data_size = imgs_trufi.shape[0] * imgs_trufi.shape[1]
		print(data_size, "data size")
		imgs_trufi = imgs_trufi.reshape((data_size, imgs_trufi.shape[2],imgs_trufi.shape[3]))
		num_tr1 = 1001
		plt.title(f"imgs_trufi_{num_tr1}")
		plt.imshow(imgs_trufi[num_tr1], cmap='gray')
		# plt.show()
		class_id_trufi = np.ones((imgs_trufi.shape[0]))
		print(class_id_trufi.shape, "class_id shape")
		num_tr2 = 2002
		plt.title(f"imgs_trufi_{num_tr2}")
		plt.imshow(imgs_trufi[num_tr2], cmap='gray')
		# plt.show()


		class_id = np.concatenate((class_id_fiesta, class_id_trufi),axis=0)
		imgs = np.concatenate((imgs_fiesta, imgs_trufi),axis=0)
		contents = np.empty(shape=(imgs.shape[0], ), dtype=np.uint32)
		imgs = np.expand_dims(imgs,axis=-1)


		# testing2
		# data_small
		num_samples = 10
		# data_med
		# num_samples = 100
		# data_large
		# num_samples = 1000
		# data_all
		# num_samples = 7000

		imgs_testing = np.concatenate((imgs_fiesta[0: num_samples],imgs_trufi[0: num_samples]), axis=0)
		class_id_testing = np.concatenate((class_id_fiesta[0: num_samples],  class_id_trufi[0: num_samples]),axis = 0)
		contents_testing = np.empty(shape=(imgs_testing.shape[0], ), dtype=np.uint32)
		imgs_testing = np.expand_dims(imgs_testing,axis=-1)

		print(imgs_testing.shape, "img testing shape")
		print(class_id_testing.shape, "class testing shape")

		testing = True

		if testing:
			print(f'max_value: {np.max(imgs_testing)}', f'min_value: {np.min(imgs_testing)}')
			imgs_testing = imgs_testing - np.min(imgs_testing)
			imgs_testing = imgs_testing / np.max(imgs_testing)
			print(f'max_value: {np.max(imgs_testing)}', f'min_value: {np.min(imgs_testing)}' )

			plt.title(f"imgs_testing no afraid")
			plt.imshow(imgs_testing[6], cmap='gray')
			plt.show()
			return imgs_testing, class_id_testing , contents_testing
		else:
			return imgs, class_id , contents








class Mnist(DataSet):

	def __init__(self):
		super().__init__()

	def read_images(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
