import os
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pickle
import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, Dense, UpSampling2D, LeakyReLU, Activation, Dropout, Flatten, MaxPooling2D, InputLayer
from keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ------------------------------------------ data-----------------------------------------------------------------------
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

        imgs_fiesta = np.array(list(fiesta_dataset.values()))
        print(imgs_fiesta.shape, "img fiest shape")
        imgs_fiesta = imgs_fiesta.transpose((0, 3,1,2))
        data_size = imgs_fiesta.shape[0] * imgs_fiesta.shape[1]
        print(data_size, "data size")
        imgs_fiesta = imgs_fiesta.reshape((data_size, imgs_fiesta.shape[2],imgs_fiesta.shape[3]))
        print(imgs_fiesta.shape, "imgs fiesta shpe")
        class_id_fiesta = np.zeros((imgs_fiesta.shape[0]))
        print(class_id_fiesta.shape, "class fiesta shape")
        num_f1 = 3001
        plt.title(f"imgs_fiesta_{num_f1}")
        plt.imshow(imgs_fiesta[num_f1], cmap='gray')
        plt.show()
        num_f2 = 2001

        for i in range(400,500, 20):
            plt.title(f"imgs__fiesta_{i}")
            plt.imshow(imgs_fiesta[i], cmap='gray')
            plt.savefig(f'/cs/casmip/nirm/embryo_project_version1/images/fiesta/imgs__fiesta_{i}.jpg')

        imgs_trufi = np.array(list(trufi_dataset.values()))
        imgs_trufi = imgs_trufi.transpose((0, 3,1,2))

        print(imgs_trufi.shape, "img trufi shape")
        data_size = imgs_trufi.shape[0] * imgs_trufi.shape[1]
        print(data_size, "data size")
        imgs_trufi = imgs_trufi.reshape((data_size, imgs_trufi.shape[2],imgs_trufi.shape[3]))
        num_tr1 = 1001
        plt.title(f"imgs_trufi_{num_tr1}")
        plt.imshow(imgs_trufi[num_tr1], cmap='gray')
        plt.show()
        class_id_trufi = np.ones((imgs_trufi.shape[0]))
        print(class_id_trufi.shape, "class_id shape")
        print(imgs_trufi.shape, "imgs trufi shpe")
        num_tr2 = 2002


        for i in range(200,300, 20):
            plt.title(f"imgs__trufi_{i}")
            plt.imshow(imgs_trufi[i], cmap='gray')
            plt.savefig(f'/cs/casmip/nirm/embryo_project_version1/images/trufi/imgs__trufi_{i}.jpg')




        class_id = np.concatenate((class_id_fiesta, class_id_trufi),axis=0)
        imgs = np.concatenate((imgs_fiesta, imgs_trufi),axis=0)
        contents = np.empty(shape=(imgs.shape[0], ), dtype=np.uint32)
        imgs = np.expand_dims(imgs,axis=-1)


        # testing
        imgs_testing = np.concatenate((np.expand_dims(imgs_fiesta[num_f1],axis = 0),np.expand_dims(imgs_fiesta[num_f2],axis = 0),  np.expand_dims(imgs_trufi[num_tr1],axis=0), np.expand_dims(imgs_trufi[num_tr2],axis=0)), axis=0)
        class_id_testing = np.array([class_id_fiesta[num_f1],class_id_fiesta[num_f2],  class_id_trufi[num_tr1], class_id_trufi[num_tr2]])
        contents_testing = np.empty(shape=(imgs_testing.shape[0], ), dtype=np.uint32)
        imgs_testing = np.expand_dims(imgs_testing,axis=-1)

        # testing2
        imgs_testing = np.concatenate((imgs_fiesta[0:7000],imgs_trufi[0:7000]), axis=0)
        class_id_testing = np.concatenate((class_id_fiesta[0:7000],  class_id_trufi[0:7000]),axis = 0)
        contents_testing = np.empty(shape=(imgs_testing.shape[0], ), dtype=np.uint32)
        imgs_testing = np.expand_dims(imgs_testing,axis=-1)

        # testing3
        num_samples = 30000
        img_new = imgs_fiesta[0: num_samples] / 255
        # img_new = np.ones_like(imgs_fiesta[0: num_samples]) / 2
        new_trufi_imgs = np.copy(imgs_trufi[0:num_samples][:,:,:]) / 255
        imgs_testing = np.concatenate((img_new, new_trufi_imgs), axis=0)
        class_id_testing = np.concatenate((class_id_fiesta[0:num_samples],  class_id_trufi[0:num_samples]),axis = 0)
        contents_testing = np.empty(shape=(imgs_testing.shape[0], ), dtype=np.uint32)

        plt.imshow(imgs_testing[0], cmap='gray')
        plt.show()
        imgs_testing = np.expand_dims(imgs_testing,axis=-1)

        print(imgs_testing.shape, "img testing shape")
        print(class_id_testing.shape, "class testing shape")
        exit()
        testing = True
        if testing:
            return imgs_testing, class_id_testing , contents_testing
        else:
            return imgs / 255, class_id , contents

# ---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    path = '/cs/casmip/nirm/embryo_project_version1/DATA_FOR_CLASSIFIER'

    data_set = embryos_dataset(path)
    x, y, _ = data_set.read_images()

    # y_new = y
    y_new = np.zeros((y.shape[0],2))
    y_new[0:int(y.shape[0] / 2),0] = 1
    y_new[int(y.shape[0] / 2):,1] = 1
    print("sample unique", y_new)
    n_samples = x.shape[0]

    n_test_samples = int(n_samples * 0.3)

    test_idx = np.random.choice(n_samples, size=n_test_samples, replace=False)
    train_idx = ~np.isin(np.arange(n_samples), test_idx)


    x_train, y_train = x[train_idx], y_new[train_idx]
    x_val, y_val = x[test_idx], y_new[test_idx]
    print("y_val",y_val)


    input_shape = [64,64,1]
    model = Sequential()
    model.add(InputLayer(input_shape))
    #
    # model.add(Conv2D(56, kernel_size=(7,7), strides=2 ))
    # # model.add(MaxPooling2D(pool_size=(2, 2) ))
    # model.add(Conv2D(112, kernel_size=(5,5), strides=2 ))
    # # model.add(MaxPooling2D(pool_size=(2, 2) ))
    # model.add(Conv2D(224, kernel_size=(5,5)))
    # # model.add(MaxPooling2D(pool_size=(2, 2) ))
    model.add(Flatten())
    # model.add(Dense(1000, activation='relu', kernel_initializer = 'random_normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='relu', kernel_initializer = 'random_normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='relu', kernel_initializer = 'random_normal'))
    # model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu', kernel_initializer = 'random_normal'))
    model.add(Dense(2, activation='softmax'))


    opt = SGD(lr = 0.001)
    model.compile(optimizer= opt,
                  loss = tensorflow.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    train = True
    if train:
        plt.title(f"testing")
        plt.imshow(np.squeeze(x_train[0]), cmap='gray')
        plt.show()
        plt.title(f"testing2")
        plt.imshow(np.squeeze(x_train[10]), cmap='gray')
        plt.show()
        print(y_train.shape, "x train shape")
        model.fit(x = x_train ,y =  y_train,epochs = 100, batch_size = 3, validation_data= (x_val, y_val), shuffle= True)
        model.save_weights(f'my_classifier_w{0}.h5')
    else:
        model.load_weights(f'my_classifier_w{0}.h5')


    print(model.predict(x_val), "predict")
    print(model.evaluate(x_val, y_val), "predict")


