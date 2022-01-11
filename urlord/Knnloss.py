import itertools
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from model.modules_unet import VGGDistance, ULordModel, UNetS, UCLordModel, UNetSC
from model.utils_unet import AverageMeter, NamedTensorDataset, ImbalancedDatasetSampler, MyRotationTransform, \
    MyRotationTransform3D
from PIL import Image
import os
import matplotlib.pyplot as plt
from os.path import join
from experiments_unet import EXP_PATH, jason_dump
from post_pro import PostProcessingSeg, postprocess_prediction
from assets import AssetManager
from torchvision import models
from config_knn_loss import config_knn
import numpy as np
from torch.nn.functional import normalize
from sklearn.decomposition import PCA


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class NetVGGFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = models.vgg16(pretrained=True)
        print(self.vggnet, "vgg summary")
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


# class VGGDistance(nn.Module):
#
#     def __init__(self, layer_ids):
#         super().__init__()
#         self.vgg = NetVGGFeatures(layer_ids)
#         self.layer_ids = layer_ids
#
#     def forward(self, batch):
#         f_batch = self.vgg(batch)
#         return f_batch


def main(base_dir, config_knn, device):
    assets = AssetManager(base_dir)
    data = np.load(assets.get_preprocess_file_path(config_knn['data_l_name']))
    data_u = np.load(assets.get_preprocess_file_path(config_knn['data_u_name']))
    data_v = np.load(assets.get_preprocess_file_path(config_knn['data_v_name']))
    data_t = np.load(assets.get_preprocess_file_path(config_knn['data_t_name']))
    ext_f = NetVGGFeatures(config_knn['perceptual_loss']['layers']).to(device)
    segs = data['segs']
    segs_u = data_u['segs']
    imgs = data['imgs']
    imgs_u = data_u['imgs']

    print("seg shape:", segs.shape)
    print("seg u shape:", segs_u.shape)

    data = dict(
        seg=torch.from_numpy(segs).permute(0, 3, 1, 2),
        img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
    )

    data_u = dict(
        seg=torch.from_numpy(segs_u).permute(0, 3, 1, 2),
        img=torch.from_numpy(imgs_u).permute(0, 3, 1, 2)
    )

    data['img'] = data['img'].type(torch.FloatTensor).to(device)
    data_u['img'] = data_u['img'].type(torch.FloatTensor).to(device)
    data['seg'] = data['seg'].type(torch.FloatTensor).to(device)
    data_u['seg'] = data_u['seg'].type(torch.FloatTensor).to(device)
    print(data['img'].size()[0], "aaa")
    ext_f.eval()
    with torch.no_grad():
        for i in range(0, data['img'].size()[0], config_knn['batch_size']):
            predictions_m_s = ext_f(data['seg'][i:i + config_knn['batch_size']])[0]
            predictions_m = ext_f(data['img'][i:i + config_knn['batch_size']])[0] * torch.round(normalize(predictions_m_s))

            if i == 0:
                predictions = predictions_m
                predictions_s = predictions_m_s
            else:
                predictions = torch.cat((predictions, predictions_m), dim=0)
                predictions_s = torch.cat((predictions_s, predictions_m_s), dim=0)

        diff = data['img'].size()[0] - predictions.size()[0]
        if diff > 0:
            s = data['img'].size()[0] - diff
            e = data['img'].size()[0]

            predictions_m_s = ext_f(data['seg'][s:e].to(device))[0]
            predictions_m = ext_f(data['img'][s:e].to(device))[0] * torch.round(normalize(predictions_m_s))

            predictions = torch.cat((predictions, predictions_m), dim=0)
            predictions_s = torch.cat((predictions_s, predictions_m_s), dim=0)

        for j in range(0, data_u['img'].size()[0], config_knn['batch_size']):
            predictions_m_u_s = ext_f(data_u['seg'][j:j + config_knn['batch_size']])[0].cpu()
            predictions_m_u = ext_f(data_u['img'][j:j + config_knn['batch_size']])[0].cpu()* torch.round(normalize(predictions_m_u_s))

            if j == 0:
                predictions_u = predictions_m_u.cpu()
                predictions_u_s = predictions_m_u_s.cpu()
            else:
                predictions_u = torch.cat((predictions_u, predictions_m_u), dim=0).cpu()
                predictions_u_s = torch.cat((predictions_u_s, predictions_m_u_s), dim=0).cpu()

        diff = data['img'].size()[0] - predictions.size()[0]
        if diff > 0:
            s = data_u['img'].size()[0] - diff
            e = data_u['img'].size()[0]
            predictions_m_u_s = ext_f(data_u['seg'][s:e].to(device))[0].cpu() * torch.round(normalize(predictions_m_u_s))
            predictions_m_u = ext_f(data_u['img'][s:e].to(device))[0].cpu()

            predictions_u = torch.cat((predictions_u, predictions_m_u), dim=0).cpu()
            predictions_u_s = torch.cat((predictions_u_s, predictions_m_u_s), dim=0).cpu()

    print(predictions_u.size(), "predictions_u size()")
    print(predictions.size(), "predictions size()")
    flatten = torch.nn.Flatten()
    # predictions = predictions.reshape((predictions.shape[0],-1))
    # predictions_u = predictions_u.reshape((predictions_u.shape[0],-1))
    predictions_u = flatten(predictions_u).cpu()
    predictions_u_s = flatten(predictions_u_s).cpu()
    predictions = flatten(predictions).cpu()
    predictions_s = flatten(predictions_s).cpu()

    print(predictions_u.size(), "predictions_u size()")
    print(predictions.size(), "predictions size()")

    U, S, V = torch.pca_lowrank(predictions.cpu())
    U_s, S_s, V_s = torch.pca_lowrank(predictions_s.cpu())

    res = torch.matmul(predictions.cpu(), V[:, :100].cpu())
    res_u = torch.matmul(predictions_u.cpu(), V[:, :100].cpu())

    res_s = torch.matmul(predictions_s.cpu(), V_s[:, :100].cpu())
    res_u_s = torch.matmul(predictions_u_s.cpu(), V_s[:, :100].cpu())

    a = list()
    # .view(b_sz, -1)
    for i in range(res_u.size()[0]):
        loss = torch.abs(res_u[i].to(device) - res.to(device)).to(device).sum(1)
        if i == 0:
            print(loss.size(), "loss size")
        loss_s = torch.abs(res_u_s[i].to(device) - res_s.to(device)).to(device).sum(1)
        min_indexes = torch.argsort(loss)
        min_indexes_s = torch.argsort(loss_s)
        # a.append(torch.abs(loss_s[min_indexes_s[0:10]] - loss[min_indexes[0:10]]).mean().item())
        print(min_indexes_s[0:10], "min indexs s")
        print(min_indexes[0:10], "min indexs")
        exit()


    a = np.array(a)
    print(np.sum(a == 1), "a equal to 1")
    print(np.sum(a == 0),  "a equal to 0")
    plt.hist(a, bins='auto')
    plt.show()
    exit()

    # print(loss, "loss")

    print(min_indexes, "loss size",loss[min_indexes[0]],loss[min_indexes[-1]])

    res = res.cpu().detach().numpy()
    res_u = res_u.cpu().detach().numpy()





    # print(res.shape, "predictions_u size()")
    # print(res_u.shape, "predictions size()")
    # exit()
    # ==================simple method =====================================================================
    # predictions = predictions.permute(0, 2, 3, 1).cpu().detach().numpy()
    # predictions_u = predictions_u.permute(0, 2, 3, 1).cpu().detach().numpy()
    #
    # print(predictions_u.shape, "predictions_u size()")
    # print(predictions.shape, "predictions size()")
    #
    # predictions_f = np.reshape(predictions,(predictions.shape[0], predictions.shape[1] * predictions.shape[2] * predictions.shape[3]))
    # predictions_u_f = np.reshape(predictions_u, (predictions_u.shape[0], predictions_u.shape[1] * predictions_u.shape[2] * predictions_u.shape[3]))

    # pca = PCA(n_components=2)
    # pred_f_pca = pca.fit(predictions_f)
    # pred_u_f_pca = pca.fit()

    # pred_f_2d = pred_f_pca.transform(predictions_f)
    # pred_u_f_2d = pred_f_pca.transform(predictions_u_f)

    plt.figure()
    plt.title("img")
    plt.plot(res_u[:, 0], res_u[:, 1], '.', color='blue')
    plt.plot(res[:, 0], res[:, 1], '.', color='green')
    plt.legend(['labeld mask, "un labeld mask"'])
    plt.show()

    plt.figure()
    plt.title("seg")
    plt.plot(res_u_s[:, 0], res_u_s[:, 1], '.', color='blue')
    plt.plot(res_s[:, 0], res_s[:, 1], '.', color='green')
    plt.legend(['labeld mask, "un labeld mask"'])
    plt.show()
    #
    # plt.figure()
    # plt.title("Repre of masks in place")
    #
    # plt.legend([])
    # # plt.show()










if __name__ == '__main__':
    base_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(base_dir, config_knn, device)
