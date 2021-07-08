import io
import numpy as np
from PIL import Image
import  matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import TensorBoard
import os
from os.path import join


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

class EvaluationCallback(TensorBoard):

	def __init__(self, model, imgs, classes, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)
		super().set_model(model.generator)

		self.__imgs = imgs
		self.__classes = classes
		self.__model_name = parts(tensorboard_dir)[-1]
		self.__model = model
		self.__n_samples_per_evaluation = 10

		print("model name:", self.__model_name)

	def on_epoch_end(self, epoch, logs={}):
		super().on_epoch_end(epoch, logs)

		img_ids = np.random.choice(self.__imgs.shape[0], size=self.__n_samples_per_evaluation, replace=False)
		imgs = self.__imgs[img_ids]
		classes = self.__classes[img_ids]
		print(classes, "class")
		print(img_ids, "img_ids")
		content_codes = self.__model.content_embedding.predict(img_ids)
		# print(content_codes, "content_codes")
		class_codes = self.__model.class_embedding.predict(classes)
		class_adain_params = self.__model.class_modulation.predict(class_codes)

		blank = np.zeros_like(imgs[0])
		blank[:,:] = 0.5
		output = [np.concatenate([blank] + [blank] + list(imgs), axis=1)]

		for i in range(self.__n_samples_per_evaluation):
			if classes[i]:
				class_id = np.ones_like(imgs[0])
			else:
				class_id = np.zeros_like(imgs[0])

			converted_imgs = [class_id]  + [imgs[i]] + [
				self.__model.generator.predict([content_codes[[j]], class_adain_params[[i]]])[0]
				for j in range(self.__n_samples_per_evaluation)
			]

			output.append(np.concatenate(converted_imgs, axis=1))

		merged_img = np.concatenate(output, axis=0)

		path_results = join('/cs/casmip/nirm/embryo_project_version1/results', self.__model_name)

		if epoch == 0:
			os.makedirs(path_results,exist_ok = True)



		print(merged_img.shape, "merged_img")
		if epoch % 100 == 0:

			image = Image.fromarray((np.squeeze(merged_img) * 255).astype(np.uint8))
			plt.title(f"results for epoch: {epoch} _ clasees {classes} ")
			plt.imshow(np.squeeze(image), cmap='gray')
			path_image = join(path_results, f' results_classes: {classes}' + '_' + f'epoch{epoch}.png')
			plt.savefig(path_image)

		# plt.savefig(f'/cs/labs/josko/nirm/embryo_project_version1/embryo_data/embryo_results/re_epoch{epoch}.png')


		summary = tf.Summary(value=[tf.Summary.Value(tag='train', image=make_image(merged_img))])
		self.writer.add_summary(summary, global_step=epoch)
		self.writer.flush()


class EncodersEvaluationCallback(TensorBoard):

	def __init__(self, model, imgs, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)
		super().set_model(model.generator)

		self.__imgs = imgs
		self.__model = model
		self.__n_samples_per_evaluation = 10

	def on_epoch_end(self, epoch, logs={}):
		if 'loss' in logs:
			logs['loss-encoders'] = logs.pop('loss')

		if 'lr' in logs:
			logs['lr-encoders'] = logs.pop('lr')

		super().on_epoch_end(epoch, logs)

		img_ids = np.random.choice(self.__imgs.shape[0], size=self.__n_samples_per_evaluation, replace=False)
		imgs = self.__imgs[img_ids]

		content_codes = self.__model.content_encoder.predict(imgs)
		class_codes = self.__model.class_encoder.predict(imgs)
		class_adain_params = self.__model.class_modulation.predict(class_codes)

		blank = np.zeros_like(imgs[0])
		output = [np.concatenate([blank] + list(imgs), axis=1)]
		for i in range(self.__n_samples_per_evaluation):
			converted_imgs = [imgs[i]] + [
				self.__model.generator.predict([content_codes[[j]], class_adain_params[[i]]])[0]
				for j in range(self.__n_samples_per_evaluation)
			]

			output.append(np.concatenate(converted_imgs, axis=1))

		merged_img = np.concatenate(output, axis=0)

		summary = tf.Summary(value=[tf.Summary.Value(tag='encoders', image=make_image(merged_img))])
		self.writer.add_summary(summary, global_step=epoch)
		self.writer.flush()


def make_image(tensor):
	height, width, channels = tensor.shape
	image = Image.fromarray((np.squeeze(tensor) * 255).astype(np.uint8))

	with io.BytesIO() as out:
		image.save(out, format='PNG')
		image_string = out.getvalue()

	return tf.Summary.Image(height=height, width=width, colorspace=channels, encoded_image_string=image_string)
