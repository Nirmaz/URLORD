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
from model.modules_unet import VGGDistance, ULordModel, UnetLatentModel, Segmentntor, ULordModel3D, UNetC
from model.utils_unet import AverageMeter, NamedTensorDataset, ImbalancedDatasetSampler, MyRotationTransform, \
    MyRotationTransform3D
from PIL import Image
# from unet2d_model import UNet
import os
import matplotlib.pyplot as plt
from os.path import join
from experiments_unet import EXP_PATH, jason_dump, EXP_PATH_C
from scipy.spatial import distance
from post_pro import PostProcessingSeg, postprocess_prediction
CUDA_LAUNCH_BLOCKING = 1
import wandb
import logging
import time
from measure import dice_func, dice_loss
from sklearn.metrics import recall_score, precision_score

def define_logger(loger_name, file_path):
    logger = logging.getLogger(loger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(file_path)
    logger.addHandler(file_handler)
    return logger


# def dice_loss(inputs, targets, smooth=1):
#     """This definition generalize to real valued pred and target vector.
#     This should be differentiable.
#     pred: tensor with first dimension as batch
#     target: tensor with first dimension as batch
#     """
#
#     # inputs = F.sigmoid(inputs)
#     # flatten label and prediction tensors
#     inputs = inputs.view(-1)
#     targets = targets.view(-1)
#     intersection = (inputs * targets).sum()
#     dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#     return 1 - dice


def parts(path):
    """
    parser to the path
    :param path:
    :return:
    """
    components = []
    while True:
        (path, tail) = os.path.split(path)
        if tail == "":
            components.reverse()
            return components
        components.append(tail)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


class UNet3D:

    def __init__(self, config=None, config_unet=None):
        super().__init__()

        self.config_unet = config_unet
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.UNet = None

    def load(self, model_dir, unet = True, num_exp=None, path_exp = None):

        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
            self.config = pickle.load(config_fd)

        with open(os.path.join(model_dir, 'config_unet.pkl'), 'rb') as config_fd:
            self.config_unet = pickle.load(config_fd)

        if num_exp != None:
            self.config['num_exp'] = num_exp
            self.config['path_exp'] = path_exp
            path_config_unet = join(path_exp, num_exp, 'config', 'config_unet.jason')
            path_config = join(path_exp, num_exp, 'config', 'config.jason')
            jason_dump(self.config, path_config)
            jason_dump(self.config_unet, path_config_unet)

        if unet:
            self.UNet = UNetC(self.config_unet)
            self.UNet.load_state_dict(torch.load(os.path.join(model_dir, 'ulord.pth')))


    def save_best_model(self, model_dir, unet = True):

        if unet:
            print("saving model......................................................")
            torch.save(self.UNet.state_dict(), os.path.join(model_dir, 'best_model.pth'))

    def save(self, model_dir, unet = True):

        with open(os.path.join(model_dir, 'config_unet.pkl'), 'wb') as config_fd:
            pickle.dump(self.config_unet, config_fd)

        with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
            pickle.dump(self.config, config_fd)

        if unet:
            print("saving model......................................................")
            torch.save(self.UNet.state_dict(), os.path.join(model_dir, 'ulord.pth'))

    def evaluate(self, path_to_weights, param_data, data_with_gt, path_result, unet3d = True):
        if unet3d:
            self.UNet.load_state_dict(torch.load(path_to_weights))
            post_processing = PostProcessingSeg(self.UNet,  data_with_gt, param_data['margin'],self.config['train']['batch_size'], param_data['patch_stride'], param_data['model_original_dim'], bb = param_data['from_bb'], unet = True)
            dict_prediction = dict()
            post_processing.predict_on_validation(dict_prediction)

            print(post_processing.max_over_lap, "post_processing.max_over_lap")
            for th in range(0, int(post_processing.max_over_lap)):

                logname_result = join(path_result, 'logging', f'unet3d_pre_th_{th}.log')
                logname_dice = f'unet3d_pre_th_{th}'
                logger_dice = define_logger(logname_dice, logname_result)
                dice_score = AverageMeter()
                pre_score = AverageMeter()
                rec_score = AverageMeter()
                for i, subject_id in enumerate(dict_prediction.keys()):
                    gt = dict_prediction[subject_id]['truth']
                    pred = dict_prediction[subject_id]['pred']
                    if i == 0:
                        print(np.unique(pred), "unique pred")

                    res = np.zeros_like(pred)
                    res[pred > th] = 1
                    res = postprocess_prediction(res)
                    dice = dice_func(gt, res)
                    res_1d = res.flatten()
                    gt_1d = gt.flatten()
                    rec = recall_score(gt_1d, res_1d)
                    pre = precision_score(gt_1d, res_1d)
                    dice_score.update(dice)
                    pre_score.update(pre)
                    rec_score.update(rec)

                    logger_dice.info(f'subject id {subject_id} dice score {dice}, recall score: {rec}, precision score: {pre} ')

                logger_dice.info(f'avg dice score {dice_score.avg} , avg recall score: {rec_score.avg}, precision score: {pre_score.avg}')

    def train_UNet3D(self, imgs, segs,classes, imgs_v, segs_v,classes_v, imgs_t, segs_t,classes_t,  model_dir, tensorboard_dir, loaded_model):

        print(f"shapes: sh_img: {imgs_t.shape[0]}, sh_seg: {segs_t.shape[0]}, sh_class: {classes_t.shape[0]}")
        print(f"shapes: sh_img: {imgs.shape[0]}, sh_seg: {segs.shape[0]}, sh_class: {classes.shape[0]}")
        print(f"shapes: sh_img: {imgs_v.shape[0]}, sh_seg: {segs_v.shape[0]}, sh_class: {classes_v.shape[0]}")
        model_name = parts(tensorboard_dir)[-1]
        path_result_train = join( self.config['path_exp'] , self.config['num_exp'], 'results', model_name, 'train')
        path_result_val = join( self.config['path_exp'] , self.config['num_exp'], 'results', model_name, 'val')
        path_result_test = join( self.config['path_exp'] , self.config['num_exp'], 'results', model_name, 'test')
        path_result_loss = join( self.config['path_exp'] , self.config['num_exp'], 'results', model_name, 'loss')

        # define logger
        logname_dice = join(self.config['path_exp'] , self.config['num_exp'], 'logging', 'unet_dice.log')


        logger_dice = define_logger('loger_dice', logname_dice)


        if not loaded_model:
            self.UNet = UNetC(self.config_unet)

        data = dict(
            img=torch.from_numpy(imgs).permute(0, 4, 3, 1, 2),
            seg=torch.from_numpy(segs).permute(0, 4, 3, 1, 2),
            class_id=torch.from_numpy(classes.astype(np.int64))
        )

        data_v = dict(
            img=torch.from_numpy(imgs_v).permute(0, 4, 3, 1, 2),
            seg=torch.from_numpy(segs_v).permute(0, 4, 3, 1, 2),
            class_id=torch.from_numpy(classes_v.astype(np.int64))
        )

        data_t = dict(
            img = torch.from_numpy(imgs_t).permute(0, 4, 3, 1, 2),
            seg=torch.from_numpy(segs_t).permute(0, 4, 3, 1, 2),
            class_id=torch.from_numpy(classes_t.astype(np.int64))
        )

        rotation_transform = MyRotationTransform3D(angles=[-90, 0, 90])
        dataset = NamedTensorDataset(data, transform = rotation_transform)
        dataset_v = NamedTensorDataset(data_v)
        dataset_t = NamedTensorDataset(data_t)
        sampler_l = ImbalancedDatasetSampler(dataset, classes, percent_list=self.config['train']['percent_list'])

        data_loader_t = DataLoader(
            dataset, sampler=sampler_l, batch_size=self.config['train']['batch_size'],
            num_workers=5, pin_memory=True
        )

        data_loader_val = DataLoader(
            dataset_v, batch_size=self.config['train']['batch_size'],
            shuffle=True, sampler=None, batch_sampler=None,
            num_workers=10, pin_memory=True, drop_last=True
        )

        data_loader_test = DataLoader(
            dataset_t, batch_size=self.config['train']['batch_size'],
            shuffle=True, sampler=None, batch_sampler=None,
            num_workers=10, pin_memory=True, drop_last=True
        )

        self.UNet.init()
        self.UNet.to(self.device)

        # criterion = VGGDistance(self.config['perceptual_loss']['layers']).to(self.device)

        optimizer = Adam([
            {
                'params': itertools.chain(self.UNet.unet.parameters()),
                'lr': self.config['train']['learning_rate']['generator']
            }
        ], betas=(0.5, 0.999))

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['train']['n_epochs'] * len(data_loader_t),
            eta_min=self.config['train']['learning_rate']['min']
        )

        # train dice loss
        train_loss = AverageMeter()
        train_loss_epoch = AverageMeter()
        dice_loss_epoch_val = AverageMeter()
        dice_loss_epoch_test = AverageMeter()
        # dice loss
        dice_loss_train = list()
        dice_loss_val = list()
        dice_loss_test = list()

        self.generate_samples_unetc(dataset, 0, path_result_train, shape=(1, imgs.shape[1], imgs.shape[2]),
                                      randomized=False)

        min_dice_loss = 1

        for epoch in range(self.config['train']['n_epochs']):
            self.UNet.train()
            train_loss.reset()
            train_loss_epoch.reset()
            pbar_t = tqdm(iterable=data_loader_t)
            pbar_val = tqdm(iterable=data_loader_val)
            pbar_test = tqdm(iterable=data_loader_test)
            start_time0 = time.time()
            for i, batch in enumerate(pbar_t):
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
                optimizer.zero_grad()
                # print(batch['img'].size(), batch['seg'].size(), "batch seg id")
                out = self.UNet(batch['img'])
                loss = dice_loss(out['mask'], batch['seg'])
                # print(f"loss{loss}")
                loss.backward()
                optimizer.step()

                if self.config['sch']:
                    scheduler.step()

                train_loss_epoch.update(loss.item())
                pbar_t.set_description_str('epoch #{}'.format(epoch))
                pbar_t.set_postfix(loss=train_loss.avg)

            pbar_t.close()

            logger_dice.info(f'epoch {epoch} # dice {train_loss_epoch.avg}')
            dice_loss_train.append(train_loss_epoch.avg)

            for batch_v in pbar_val:
                # print("here")
                self.UNet.eval()
                batch_v = {name: tensor.to(self.device) for name, tensor in batch_v.items()}
                out = self.UNet(batch_v['img'])
                loss_val = dice_loss(out['mask'], batch_v['seg'])
                dice_loss_epoch_val.update(loss_val.item())

            for batch_t in pbar_test:
                # print("here")
                self.UNet.eval()
                batch_t = {name: tensor.to(self.device) for name, tensor in batch_t.items()}
                out = self.UNet(batch_t['img'])
                loss_test = dice_loss(out['mask'], batch_t['seg'])
                dice_loss_epoch_test.update(loss_test.item())

            # dice_txt.write('val: ' +  f'# {dice_loss_epoch_val.avg}')
            logger_dice.info(f'val epoch {epoch} # dice {dice_loss_epoch_val.avg}')
            logger_dice.info(f'test epoch {epoch} # dice {dice_loss_epoch_test.avg}')

            dice_loss_val.append(dice_loss_epoch_val.avg)
            dice_loss_test.append(dice_loss_epoch_test.avg)

            if dice_loss_epoch_val.avg < min_dice_loss:
                min_dice_loss = dice_loss_epoch_val.avg
                self.save_best_model(model_dir , unet = True)

            if min_dice_loss:
                os.makedirs(path_result_loss, exist_ok=True)

            if epoch % 5 == 0:
                x = np.arange(epoch + 1)
                plt.figure()
                plt.plot(x, dice_loss_train)
                plt.plot(x, dice_loss_val)
                plt.plot(x, dice_loss_test)
                plt.title(f'Loss vs. epochs {epoch}')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Training', 'Validation', 'test'], loc='upper right')
                plt.savefig(join(path_result_loss, f'loss_for_epoch{epoch}.jpeg'))
                print('done')

            print("------------ %s seg time alll -------------------------------" % (time.time() - start_time0))
            self.save(model_dir, unet = True)


            self.generate_samples_unetc(dataset, epoch, path_result_train,
                                                             shape=(1, imgs.shape[1], imgs.shape[2]), randomized=False)

            self.generate_samples_unetc(dataset, epoch, path_result_train,
                                                              shape=(1, imgs.shape[1], imgs.shape[2]), randomized=True)

            self.generate_samples_unetc(dataset_v, epoch, path_result_val,
                                                             shape=(1, imgs.shape[1], imgs.shape[2]), randomized=False)

            self.generate_samples_unetc(dataset_v, epoch, path_result_val,
                                                              shape=(1, imgs.shape[1], imgs.shape[2]), randomized=True)

            self.generate_samples_unetc(dataset_t, epoch, path_result_test,
                                        shape=(1, imgs.shape[1], imgs.shape[2]), randomized=False)

            self.generate_samples_unetc(dataset_t, epoch, path_result_test,
                                        shape=(1, imgs.shape[1], imgs.shape[2]), randomized=True)



    def generate_samples_unetc(self, dataset, epoch, path_result, shape, n_samples = 5, randomized=False):
        if randomized:
            rnd = 'random'
        else:
            rnd = 'not_random'

        self.UNet.eval()
        if randomized:
            random = np.random
        else:
            random = np.random.RandomState(seed=1234)

        img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))
        samples = dataset[img_idx]
        samples = {name: tensor.to(self.device) for name, tensor in samples.items()}

        path_results_output = join(path_result, 'output')
        path_results_anatomy = join(path_result, 'anatomy')
        path_results_dice = join(path_result, 'dice')

        output_dice = list()
        output_dice_th = list()

        # the number of slice to take from the event
        num_slice = 10
        if epoch == 0:
            os.makedirs(path_results_output, exist_ok=True)
            os.makedirs(path_results_anatomy, exist_ok=True)
            os.makedirs(path_results_dice, exist_ok=True)

        cloned_all_samples = torch.clone(samples['img'])
        # print(cloned_all_samples.size(), "cloned all samples first step.............................................")
        cloned_all_samples = torch.squeeze(cloned_all_samples, dim=1)
        cloned_all_samples = cloned_all_samples[:, num_slice, :, :]
        # print(cloned_all_samples.size(), "cloned all samples next step.............................................")

        blank = torch.ones(shape)
        blank[:, :] = 0.5
        # build output
        output = list()

        output1 = [blank.detach().cpu()]
        classes = np.zeros((n_samples))
        for k in range(n_samples):
            if samples['class_id'][[k]]:
                classes[k] = 1
                class_id = torch.ones(shape)
            else:
                class_id = torch.zeros(shape)

            # print(class_id.size, "len output 1")
            output1.append(class_id.detach().cpu())

        # print(output1, "output")
        output.append(torch.cat(output1, dim=2))

        # add samples
        output1 = [blank.detach()]
        classes = np.zeros((n_samples))
        for k in range(n_samples):
            # print(torch.unsqueeze(cloned_all_samples[k], dim=0).size(), "cloned")
            output1.append(torch.unsqueeze(cloned_all_samples[k], dim=0).detach().cpu())
        output.append(torch.cat(output1, dim=2))



        cloned_all_samples_seg = torch.clone(samples['seg'])
        cloned_all_samples_seg = torch.squeeze(cloned_all_samples_seg, dim=1)
        cloned_all_samples_seg = cloned_all_samples_seg[:, num_slice, :, :]

        output1 = [blank.detach().cpu()]
        classes = np.zeros((n_samples))
        for k in range(n_samples):
            # print(torch.unsqueeze(cloned_all_samples_seg[k], dim=0).size(), "cloned")
            output1.append(torch.unsqueeze(cloned_all_samples_seg[k], dim=0).detach().cpu())
        output.append(torch.cat(output1, dim=2))

        converted_imgs = [blank]
        for i in range(n_samples):
            out = self.UNet(samples['img'][[i]])
            mask = torch.round(out['mask'][0])
            converted_imgs.append(torch.unsqueeze(mask[0][num_slice], 0).detach().cpu())
            output_dice.append(dice_loss(torch.unsqueeze(mask[0][num_slice], 0),
                                         torch.unsqueeze(samples['seg'][[i]][0][0][num_slice], 0)))

        output.append(torch.cat(converted_imgs, dim=2))

        converted_imgs = [blank.detach().cpu()]
        for i in range(n_samples):
            out = self.UNet(samples['img'][[i]])
            mask = out['mask'][0]
            converted_imgs.append(torch.unsqueeze(mask[0][num_slice], 0).detach().cpu())
            output_dice_th.append(dice_loss(torch.unsqueeze(mask[0][num_slice], 0),
                                            torch.unsqueeze(samples['seg'][[i]][0][0][num_slice], 0)))

        output.append(torch.cat(converted_imgs, dim=2))

        if epoch % 2 == 0:
            output_img = torch.cat(output, dim=1).cpu().detach().numpy()
            output_img = np.squeeze(output_img)
            merged_img = Image.fromarray((output_img * 255).astype(np.uint8))
            plt.figure()
            path_image = join(path_results_output,
                              f' results_classes: {classes}' + '_' + f'epoch{epoch}' + '_' + rnd + '.png')
            plt.title(f"epoch number: {epoch}")
            plt.imshow(merged_img, cmap='gray')
            plt.savefig(path_image)
        # plt.show()

        # plt.show()
        if epoch % 2 == 0:
            output_dice = torch.tensor(output_dice).cpu().detach().numpy()
            x = np.arange(n_samples)
            path_image = join(path_results_dice,
                              f' results_classes: {classes}' + '_' + f'epoch{epoch}' + '_' + rnd + '.png')
            plt.figure()
            plt.title(f"epoch number: {epoch}")
            plt.bar(x, output_dice)
            plt.savefig(path_image)


