import os
import torch
from abc import ABC, abstractmethod
import os
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.data.dataset import Dataset
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering

# load GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CUDA_LAUNCH_BLOCKING = 1


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)




def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

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



class DataSet(ABC):

    def __init__(self, base_dir=None):
        super().__init__()
        self._base_dir = base_dir

    @abstractmethod
    def read_images(self):
        pass


class NamedTensorDataset(Dataset):

	def __init__(self, named_tensors):
		assert all(list(named_tensors.values())[0].size(0) == tensor.size(0) for tensor in named_tensors.values())
		self.named_tensors = named_tensors

	def __getitem__(self, index):
		return {name: tensor[index] for name, tensor in self.named_tensors.items()}

	def __len__(self):
		return list(self.named_tensors.values())[0].size(0)

	def subset(self, indices):
		return NamedTensorDataset(self[indices])


class embryos_dataset(DataSet):

    def __init__(self, base_dir):
        super().__init__(base_dir)

    def read_images(self, size_jump):
        show_images = True
        # load data set
        with open(os.path.join(self._base_dir, 'placenta', 'fetal_data_patches.h5'), "rb") as opened_file:
            fiesta_dataset = pickle.load(opened_file)

        with open(os.path.join(self._base_dir, 'TRUFI', 'fetal_data_patches.h5'), "rb") as opened_file:
            trufi_dataset = pickle.load(opened_file)

        with open(os.path.join(self._base_dir, 'placenta', 'fetal_data_patches_gt.h5'), "rb") as opened_file:
            fiesta_dataset_gt = pickle.load(opened_file)

        with open(os.path.join(self._base_dir, 'TRUFI', 'fetal_data_patches_gt.h5'), "rb") as opened_file:
            trufi_dataset_gt = pickle.load(opened_file)

        #  load fiest
        imgs_fiesta = np.array(list(fiesta_dataset.values()))

        imgs_fiesta_gt = np.array(list(fiesta_dataset_gt.values()))
        class_id_fiesta = np.zeros((imgs_fiesta.shape[0]))
        print(imgs_fiesta.shape, "imgs_fiesta.shape")
        print(imgs_fiesta_gt.shape, "imgs_fiesta_gt.shape")
        if show_images:
            num_f1 = 2001
            for i in range(3):
                plt.title(f"imgs_fiesta_{num_f1 + i}")
                plt.imshow(imgs_fiesta[num_f1 + i], cmap='gray')
                plt.show()

        # load trufi
        imgs_trufi = np.array(list(trufi_dataset.values()))

        imgs_trufi_gt = np.array(list(trufi_dataset_gt.values()))
        class_id_trufi = np.ones((imgs_trufi.shape[0]))
        print(imgs_trufi.shape, "imgs_trufi.shape")
        print(imgs_trufi_gt.shape, "imgs_trufi_gt.shape")
        if show_images:
            num_t1 = 2001
            for i in range(3):
                plt.title(f"imgs_trufi_{num_t1 + i}")
                plt.imshow(imgs_trufi[num_t1 + i], cmap='gray')
                plt.show()

        # concatnate all
        class_id = np.concatenate((class_id_fiesta, class_id_trufi), axis=0)
        imgs = np.concatenate((imgs_fiesta, imgs_trufi), axis=0)
        segs = np.concatenate((imgs_fiesta_gt, imgs_trufi_gt), axis=0)
        imgs = np.expand_dims(imgs, axis=-1)

        # concatante part
        print(imgs_fiesta[::size_jump].shape, "imgs_fiesta[: :self.size_jump]")
        print(imgs_trufi[::size_jump].shape, "imgs_trufi[: :self.size_jump]")
        print(imgs_fiesta_gt[::size_jump].shape, "imgs_fiesta_gt[::size_jump]")
        print(imgs_trufi_gt[::size_jump].shape, "imgs_trufi_gt[: :self.size_jump]")

        imgs_fiesta_part = imgs_fiesta[::size_jump]
        class_id_fiesta_part = class_id_fiesta[::size_jump]
        imgs_fiesta_gt_part = imgs_fiesta_gt[::size_jump]

        imgs_trufi_part = imgs_trufi[::size_jump]
        class_id_trufi_part = class_id_trufi[::size_jump]
        imgs_trufi_gt_part = imgs_trufi_gt[::size_jump]

        min_size = np.min((imgs_trufi_part.shape[0], imgs_fiesta_part.shape[0]))
        print(min_size, "min size")

        imgs_part = np.concatenate((imgs_fiesta_part[: min_size], imgs_trufi_part[: min_size]), axis=0)
        segs_part = np.concatenate((imgs_fiesta_gt_part[: min_size], imgs_trufi_gt_part[: min_size]), axis=0)
        class_id_part = np.concatenate((class_id_fiesta_part[0: min_size], class_id_trufi_part[0: min_size]), axis=0)
        class_id_part =  np.expand_dims(class_id_part, axis=-1)
        imgs_part = np.expand_dims(imgs_part, axis=-1)
        segs_part = np.expand_dims(segs_part, axis=-1)

        # normlaize data
        imgs_part = imgs_part - np.min(imgs_part)
        imgs_part = imgs_part / np.max(imgs_part)

        # if show_images:
        img_part = 0
        for i in range(3):
            # plt.title(f"imgs_trufi_{img_part + i}")
            plt.imshow(imgs_part[img_part + i], cmap='gray')
            plt.show()
            plt.imshow(segs_part[img_part + i], cmap='gray')

            plt.show()

        imgs = imgs - np.min(imgs)
        imgs = imgs / np.max(imgs)

        PART = True
        if PART:
            return imgs_part, class_id_part, segs_part

        else:
            return imgs, class_id, segs


model = torch.nn.Sequential(

    torch.nn.Flatten(),
    torch.nn.Dropout(p = 0.2),
    torch.nn.Linear(128*128,512, bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p = 0.5),
    torch.nn.Linear(512,256, bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p = 0.5),
    torch.nn.Linear(256,128, bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p = 0.5),
    torch.nn.Linear(128, 2, bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(2,1, bias=True),
    torch.nn.Sigmoid()
).to(device)






if __name__ == '__main__':
    a = torch.tensor([1,1,1,0,0,1,0,1])
    print(a)
    a = a.repeat_interleave(20)
    print(a)
    exit()
 
 
 
 



 

    print(torch.cuda.is_available(), " torch.cuda.is_available() ")

    path = '/cs/casmip/nirm/embryo_project_version1/DATA_NEWLS1'

    data_set = embryos_dataset(path)
    imgs, classes, _ = data_set.read_images(1)



    n_samples = imgs.shape[0]

    n_test_samples = int(n_samples * 0.3)
    print(n_test_samples, "n_test_samples")
    test_idx = np.random.choice(n_samples, size = n_test_samples, replace=False)
    train_idx = ~np.isin(np.arange(n_samples), test_idx)

    data_train = dict(
        img = torch.from_numpy(imgs[train_idx]).permute(0, 3, 1, 2),
        label = torch.from_numpy(classes[train_idx])
    )
    dataset_train = NamedTensorDataset(data_train)

    # print("arrive here 2")
    data_validation = dict(
        img=torch.from_numpy(imgs[test_idx]).permute(0, 3, 1, 2),
        label=torch.from_numpy(classes[test_idx])
    )
    dataset_val = NamedTensorDataset(data_validation)

    data_loader_train = DataLoader(
        dataset_train, batch_size = 64,
        shuffle = True, sampler = None, batch_sampler=None,
        num_workers = 1, pin_memory = True, drop_last=True
    )

    data_loader_val = DataLoader(
        dataset_val, batch_size=64,
        shuffle=False, sampler=None, batch_sampler=None,
        num_workers=1, pin_memory=True, drop_last=True
    )


    num_epoch = 500

    loss_function = torch.nn.BCELoss().to(device)

    # loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    train_loss = AverageMeter()

    train = True

    if train:

        for epoch in range(num_epoch):
            model.train()
            pbar = tqdm(iterable = data_loader_train)

            for batch in tqdm(pbar):

                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                # print(batch)

                optimizer.zero_grad()
                # print(batch['img'].shape, "shape")
                out = model(batch['img'].float())

                loss = loss_function(out, batch['label'].float())


                loss.backward()
                optimizer.step()

                train_loss.update(loss.item())
                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(loss = train_loss.avg)

            pbar.close()

            print(f"epoch:{epoch} ,training loss: {train_loss.avg}")


            model.eval()
            train_loss_val = AverageMeter()
            pbar_val = tqdm(iterable=data_loader_val)

            for batch in pbar_val:

                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                with torch.no_grad():

                    out = model(batch['img'].float())
                    results = torch.cat((batch['label'],out), dim= 1)
                    if epoch % 10 == 0:
                        print(results, "results")
                    loss = loss_function(out, batch['label'].float())

                    train_loss_val.update(loss.item())
                    pbar.set_description_str('epoch #{}'.format(epoch))
                    pbar.set_postfix(loss = train_loss.avg)

            print(f"epoch:{epoch} ,validation loss: {train_loss_val.avg}")

        torch.save(model.state_dict(), '/cs/casmip/nirm/embryo_project_version1/classfier_weights/model_weights2b.pth')

    else:
        model.load_state_dict(torch.load('/cs/casmip/nirm/embryo_project_version1/classfier_weights/model_weights2b.pth'))
        model.eval()

        for n, m in model.named_modules():
            print(n, "n")
            if n == "8":
                m.register_forward_hook(get_features('5'))



        PREDS = []
        FEATS = []
        LABELS = []
        for i, batch in enumerate(data_loader_val):
            # move to device
            batch = {name: tensor.to(device) for name, tensor in batch.items()}

            # placeholder for batch features
            features = {}

            # forward pass [with feature extraction]
            preds = model(batch['img'].float())

            # add feats and preds to lists
            print(features['5'].cpu().numpy().shape, " feature shape")
            PREDS.append(preds.detach().cpu().numpy())
            FEATS.append(features['5'].cpu().numpy())
            LABELS.append(batch['label'].cpu().numpy())

        PREDS = np.concatenate(PREDS)
        FEATS = np.concatenate(FEATS)
        LABELS = np.concatenate(LABELS)
        print('- preds shape:', PREDS.shape)
        print('- feats shape:', FEATS.shape)

        print(LABELS.shape, "FEATS shape")

        fiesta_activations = FEATS[np.squeeze(LABELS) == 0]
        trufi_activations = FEATS[np.squeeze(LABELS) == 1]




        print(fiesta_activations.shape, "fiesta_activations")
        print(trufi_activations.shape, "trufi_activations")
        print(f'trufi max_value: {np.max(trufi_activations)}', f'trufi min_value: {np.min(trufi_activations)}')
        print(f'fiesta max_value: {np.max(fiesta_activations)}', f'fiesta min_value: {np.min(fiesta_activations)}')


        fiesta_activations_min = np.min(fiesta_activations, axis = 0)
        fiesta_activations = np.subtract(fiesta_activations, fiesta_activations_min)
        fiesta_activations_max_val = np.max(fiesta_activations, axis = 0 )
        fiesta_activations = np.divide(fiesta_activations ,fiesta_activations_max_val)
        fiesta_activations_mean = np.mean(fiesta_activations, axis=0)
        print(fiesta_activations_mean, "fiesta ")

        trufi_activations_min = np.min(trufi_activations, axis = 0)
        trufi_activations = np.subtract(trufi_activations, trufi_activations_min)
        trufi_activations_max_val = np.max(trufi_activations, axis = 0 )
        trufi_activations = np.divide(trufi_activations ,trufi_activations_max_val)
        trufi_activations_mean = np.mean(trufi_activations, axis=0)
        print(trufi_activations_mean, "trufi ")
        c_e_d = dict()
        c_e_d['1'] = trufi_activations_mean
        c_e_d['0'] = fiesta_activations_mean
        out_file = os.path.join('/cs/casmip/nirm/embryo_project_version1/class_embedding', 'emb_dict.json')
        pickle_dump(c_e_d, out_file)
        file = pickle_load(out_file)
        print(file)
        exit()

        X_embedded = TSNE(n_components=2).fit_transform(FEATS)
        # kmeans = KMeans(n_clusters=2).fit(FEATS)
        kmeans = SpectralClustering(n_clusters=2).fit(FEATS)

        plt.figure()
        plt.title("Features for Trufi/Fiesta protocols")

        print(LABELS.shape, "labels shape")
        plt.plot(X_embedded[np.squeeze(LABELS) == 0,0], X_embedded[np.squeeze(LABELS) == 0, 1], '.', color='green')
        plt.plot(X_embedded[np.squeeze(LABELS) == 1,0], X_embedded[np.squeeze(LABELS) == 1, 1], '.', color='blue')
        plt.legend(['Fiesta', "TRUFI"])
        plt.show()
        # plt.savefig("trufi_fiesta.jpg")

        plt.figure()
        plt.title("Features for Trufi/Fiesta protocols Kmean")

        print(LABELS.shape, "labels shape")
        plt.plot(X_embedded[kmeans.labels_ == 0,0], X_embedded[kmeans.labels_ == 0, 1], '.', color='orange')
        plt.plot(X_embedded[kmeans.labels_ == 1,0], X_embedded[kmeans.labels_ == 1, 1], '.', color='red')
        plt.legend(['0', "1"])
        plt.show()
        # plt.savefig("trufi_fiesta_kmeans.jpg")


