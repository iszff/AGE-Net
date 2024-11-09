# add fc compared with v1
from data import read_diap_with_depth
import json
from models import adaptive_wing_loss
import models
import time
import logging
import fire
import numpy as np
import utils.vispyplot as vplt
import visdom
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils import data
import torch as th
from utils.trainer import Trainer
import os

from matplotlib.pyplot import tricontour
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

th.backends.cudnn.deterministic = False
th.backends.cudnn.benchmark = True
# search and decide faster conv algorithm


def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


class GazeTrainer(Trainer):
    def __init__(self,
                 # data parameters
                 data_root="data_root_default",\
                 batch_size_train=3,
                 batch_size_val=4,
                 num_workers=0,
                 # trainer parameters
                 is_cuda=True,
                 exp_name="gaze_eyediap_v1",
                 result_root='/home/workspace/your_path_save_training_result',
                 test_choose=0
                 ):
        #super(GazeTrainer, self).__init__(checkpoint_dir=result_save_log_path+'checkpoint'+ exp_name, is_cuda=is_cuda)
        super(GazeTrainer, self).__init__(
            checkpoint_dir=result_root+"/checkpoint", is_cuda=is_cuda)
 
        self.data_root = data_root
        self.result_root = result_root + exp_name

        logging.info(
            f'======================{exp_name}.py ===================')

        logging.info('train batchsize: {}'.format(batch_size_train))
        logging.info('test batchsize: {}'.format(batch_size_val))
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers

        self.eyediap_image_path = '/data/Eyediap_3D_face_depth/Image'
        self.eyediap_label_path = '/data/Eyediap_3D_face_depth/Label/ClusterLabel'
        # self.eyediap_image_path = '/data/Eyediap_3D_face_depth_minor_for_debug/Image'
        # self.eyediap_label_path = '/data/Eyediap_3D_face_depth_minor_for_debug/Label'

        self.total_pred_mse_loss_coord_train_for_compare_record = {}
        self.left_mse_loss_coord_train_for_compare_record = {}
        self.right_mse_loss_coord_train_for_compare_record = {}
        self.total_pred_mse_loss_coord_val_for_compare_record = {}
        self.left_mse_loss_coord_val_for_compare_record = {}
        self.right_mse_loss_coord_val_for_compare_record = {}

        self.total_pred_train_angular_precison_record = {}
        self.left_train_angular_precison_record = {}
        self.right_train_angular_precison_record = {}
        self.train_loss_yaw_pitch_record = {}
        #self.right_train_loss_yaw_pitch_record = {}

        self.total_pred_val_angular_precison_record = {}
        self.left_val_angular_precison_record = {}
        self.right_val_angular_precison_record = {}
        self.val_loss_yaw_pitch_record = {}
        # self.right_val_loss_yaw_pitch_record = {}

        folder = os.listdir(self.eyediap_label_path)
        folder.sort()
        testindex = test_choose
        test = folder.pop(testindex)
        trains = folder
        self.trainlabelpath = [os.path.join(
            self.eyediap_label_path, j) for j in trains]
        self.testlabelpath = os.path.join(self.eyediap_label_path, test)

        self.prec_loss_val_record_file = os.path.join(result_root, experiment_name+"_test"+str(
            testindex) + "_tbatchsize" + str(batch_size_train) + '_metric.log')
        #self.prec_loss_val_record_file = os.path.join(result_root, experiment_name+"_test"+str(testindex) +"_tbatchsize"+ str(batch_size_train)+ '_metric_right.log')

        # initialize models
        self.exp_name = exp_name
        model = models.__dict__[exp_name]  
        # import pudb
        # pudb.set_trace()
        self.models.resnet = model.resnet34(pretrained=True)
        self.models.decoder = model.Decoder()
        # self.models.depth_loss = model.DepthBCE(0)
        # self.models.refine_depth = model.RefineDepth()

        self.adaptive_wing_loss = adaptive_wing_loss.AdaptiveWingLoss()
        self.weights_init(self.models.decoder)  #  don't work
        # self.weights_init(self.models.refine_depth)

        # initialize extra variables
        self.extras.best_loss_base_val = 99999
        self.extras.best_loss_refine_val = 99999
        self.extras.last_epoch_headpose = -1
        self.extras.last_epoch_base = -1

        # initialize meter variables
        self.meters.left_mse_loss_coord_train_for_compare = {}
        self.meters.right_mse_loss_coord_train_for_compare = {}
        self.meters.total_pred_mse_loss_coord_train_for_compare = {}
        self.meters.left_mse_loss_coord_val_for_compare = {}
        self.meters.right_mse_loss_coord_val_for_compare = {}
        self.meters.total_pred_mse_loss_coord_val_for_compare = {}

        self.meters.loss_coord_train = {}
        self.meters.loss_depth_train = {}
        self.meters.loss_coord_val = {}
        self.meters.loss_depth_val = {}
        self.meters.left_prec_coord_train = {}
        self.meters.right_prec_coord_train = {}
        self.meters.total_pred_prec_coord_train = {}
        self.meters.left_prec_coord_val = {}
        self.meters.right_prec_coord_val = {}
        self.meters.total_pred_prec_coord_val = {}

        # initialize visualizing
        #visdom.Visdom(server='http://localhost', port=8097)
        vplt.vis_config.server = 'http://10.10.10.100'
        vplt.vis_config.port = 5000
        vplt.vis_config.env = exp_name

        pth_load_dir = "no_/home/workspace/your_save_path/SaveResult/2022-05-31-05-07gaze_eyediap_v2_depth3_fc_lrclus2test_batchsize4/checkpoint"
        # "/home/workspace/your_save_path/SaveResult/2022-05-31-05-31gaze_eyediap_v2_depth3_fc_lrclus1test_batchsize4/checkpoint"
        load_epoch = "epoch_73.pth.tar"
        if os.path.isfile(os.path.join(pth_load_dir, load_epoch)):
            logging.info('=======================resume ===================')
            load_root = os.path.join(pth_load_dir, load_epoch)
            self.logger.info(f"load checkpoint @ {load_root}")
            self.load_state_dict(load_root)



    def train_base(self, epochs, lr=1e-4, use_refined_depth=False, fine_tune_headpose=True):
        # prepare logger
        self.temps.base_logger = self.logger.getChild('train_base')
        self.temps.base_logger.info('preparing for base training loop.')
        logging.info('preparing for base training loop.')

        # prepare dataloader
        self.temps.train_loader = self._get_eyediap_trainloader()
        self.temps.val_loader = self._get_eyediap_valloader()
        self.temps.num_iters = len(self.temps.train_loader)

        self.temps.lr = lr
        self.temps.epochs = epochs

        self.temps.use_refined_depth = use_refined_depth
        self.temps.fine_tune_headpose = fine_tune_headpose
        # start training loop
        self.temps.epoch = self.extras.last_epoch_base
        self.temps.base_logger.info(
            f'start base training loop @ epoch {self.extras.last_epoch_base + 1}.')
        logging.info(
            f'start base training loop @ epoch {self.extras.last_epoch_base + 1}.')
        for epoch in range(self.extras.last_epoch_base + 1, epochs):
            if(epoch < 20):
                self.temps.lr = 1e-2
            elif(epoch < 30):
                self.temps.lr = 1e-3
            else:
                self.temps.lr = 1e-4

            self.temps.epoch = epoch
            # initialize meters for new epoch
            self._init_base_meters()
            # train one epoch
            logging.info('=======================train===================')

            self._train_base_epoch()
            # test on validation set
            logging.info('=======================test===================')
            self._test_base()
            # save checkpoint
            self._log_base()
            self.extras.last_epoch_base = epoch
            logging.info(
                '=====================save checkpoint=================== ')

            self.save_state_dict(f'epoch_{epoch}.pth.tar')
            self.save_state_dict(f'epoch_latest.pth.tar')
            # plot result
            # self._plot_base()
            # logging
            # self._log_base()

            # save_checkpoint(
            # {
            #     'epoch': epoch,
            #     'state_dict': model.state_dict(),
            #     'minimal_error': minimal_error,
            #     'optimizer': optimizer.state_dict(),
            # }, os.path.join(ckpt_path, 'model_best.pth.tar'))

        # cleaning
        self.models.resnet.cpu()
        self.models.decoder.cpu()
        del self.temps.train_loader
        del self.temps.val_loader

        return self

    def train_headpose(self, epochs, lr=2e-4, lambda_loss_mse=1):
        self.temps.lambda_loss_mse = lambda_loss_mse

        os.makedirs(os.path.join(self.result_root,
                    "train", "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.result_root,
                    "val", "depth"), exist_ok=True)

        # prepare logger
        self.temps.headpose_logger = self.logger.getChild('train_headpose')
        self.temps.headpose_logger.info(
            'preparing for headpose training loop.')

        # prepare dataloader
        self.temps.train_loader = self._get_eyediap_trainloader()
        self.temps.val_loader = self._get_eyediap_valloader()
        self.temps.num_iters = len(self.temps.train_loader)
        self.temps.epochs = epochs
        self.temps.lr = lr

        # start training loop
        self.temps.epoch = self.extras.last_epoch_headpose
        self.temps.headpose_logger.info(
            'start headpose training loop @ epoch {}.'.format(self.extras.last_epoch_headpose + 1))
        for epoch in range(self.extras.last_epoch_headpose + 1, epochs):
            self.temps.epoch = epoch
            # initialize meters for new epoch
            self._init_headpose_meters()
            # train one epoch
            self._train_headpose_epoch()
            # test on validation set
            self._test_headpose()
            # save checkpoint
            self.extras.last_epoch_headpose = epoch
            self.save_state_dict(f'epoch_{epoch}.pth.tar')
            self.save_state_dict(f'epoch_latest.pth.tar')

            # plot result
            # self._plot_headpose()
            # logging
            self._log_headpose()

        # cleaning
        self.models.refine_depth.cpu()
        del self.temps.train_loader
        del self.temps.val_loader

        return self

    def resume(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.isfile(path):
            self.load_state_dict(filename)
            self.logger.info('load checkpoint from {}'.format(path))
        return self

    def _prepare_model(self, model, train=True):
        if self.is_cuda:
            model.cuda()
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
        if train:
            model.train()
        else:
            model.eval()
        return model

    def _get_eyediap_trainloader(self):
        logging.info('training data ')
        imagepath = self.eyediap_image_path
        trainlabelpath = self.trainlabelpath
        train_loader = read_diap_with_depth.txtload(trainlabelpath, imagepath, self.batch_size_train,
                                                    shuffle=True, num_workers=self.num_workers, header=True)

        return train_loader

    def _get_eyediap_valloader(self):
        logging.info('testf data ')
        imagepath = self.eyediap_image_path
        testlabelpath = self.testlabelpath
        test_loader = read_diap_with_depth.txtload(testlabelpath, imagepath, self.batch_size_train,
                                                   shuffle=True, num_workers=self.num_workers, header=True)
        return test_loader

    def _get_trainloader(self):
        logger = self.logger
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])
        }

        transformed_train_dataset = GazePointAllDataset(root_dir=self.data_root,
                                                        transform=data_transforms['train'],
                                                        phase='train',
                                                        face_image=True, face_depth=True, eye_image=True,
                                                        eye_depth=True,
                                                        info=True, eye_bbox=True, face_bbox=True, eye_coord=True)
        logger.info('The size of training data is: {}'.format(
            len(transformed_train_dataset)))
        train_loader = data.DataLoader(transformed_train_dataset, batch_size=self.batch_size_train, shuffle=True,
                                       num_workers=self.num_workers)

        return train_loader

    def _get_valloader(self):
        logger = self.logger
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])
        }

        transformed_test_dataset = GazePointAllDataset(root_dir=self.data_root,
                                                       transform=data_transforms['val'],
                                                       phase='val',
                                                       face_image=True, face_depth=True, eye_image=True, eye_depth=True,
                                                       info=True, eye_bbox=True, face_bbox=True, eye_coord=True)

        logger.info('The size of testing data is: {}'.format(
            len(transformed_test_dataset)))

        test_loader = data.DataLoader(transformed_test_dataset, batch_size=self.batch_size_val, shuffle=False,
                                      num_workers=self.num_workers)
        return test_loader

    def _init_base_meters(self):
        epoch = self.temps.epoch
        self.meters.left_mse_loss_coord_train_for_compare[epoch] = []
        self.meters.right_mse_loss_coord_train_for_compare[epoch] = []
        self.meters.total_pred_mse_loss_coord_train_for_compare[epoch] = []

        self.meters.left_mse_loss_coord_val_for_compare[epoch] = 0
        self.meters.right_mse_loss_coord_val_for_compare[epoch] = 0
        self.meters.total_pred_mse_loss_coord_val_for_compare[epoch] = 0

        self.meters.loss_coord_train[epoch] = []
        self.meters.loss_coord_val[epoch] = 0
        self.meters.left_prec_coord_train[epoch] = []
        self.meters.right_prec_coord_train[epoch] = []
        self.meters.total_pred_prec_coord_train[epoch] = []
        self.meters.left_prec_coord_val[epoch] = 0
        self.meters.right_prec_coord_val[epoch] = 0
        self.meters.total_pred_prec_coord_val[epoch] = 0

    def _init_headpose_meters(self):
        epoch = self.temps.epoch
        self.meters.loss_depth_train[epoch] = []
        self.meters.loss_depth_val[epoch] = 0

    def _plot_base(self):
        with vplt.set_draw(name='loss_base') as ax:
            ax.plot(list(self.meters.loss_coord_train.keys()),
                    np.mean(list(self.meters.loss_coord_train.values()), axis=1), label='loss_coord_train')
            ax.plot(list(self.meters.loss_coord_val.keys()),
                    list(self.meters.loss_coord_val.values()), label='loss_coord_val')
            ax.set_title('loss base')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_xlim(0, self.temps.epochs)
            ax.legend()
            ax.grid(True)
        with vplt.set_draw(name='prec_base') as ax:
            ax.plot(list(self.meters.prec_coord_train.keys()),
                    np.mean(list(self.meters.prec_coord_train.values()), axis=1), label='prec_coord_train')
            ax.plot(list(self.meters.prec_coord_val.keys()),
                    list(self.meters.prec_coord_val.values()), label='prec_coord_val')
            ax.set_title('prec base')
            ax.set_xlabel('epoch')
            ax.set_ylabel('prec')
            ax.set_xlim(0, self.temps.epochs)
            ax.legend()
            ax.grid(True)

    def _plot_headpose(self):
        with vplt.set_draw(name='loss_depth') as ax:
            ax.plot(list(self.meters.loss_depth_train.keys()),
                    np.mean(list(self.meters.loss_depth_train.values()), axis=1), label='loss_depth_train')
            ax.plot(list(self.meters.loss_depth_val.keys()),
                    list(self.meters.loss_depth_val.values()), label='loss_depth_val')
            ax.set_title('loss depth')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_xlim(0, self.temps.epochs)
            ax.legend()
            ax.grid(True)

    def _log_base(self):
        epoch = self.temps.epoch,
        left_mse_loss_coord_train_for_compare = np.mean(
            self.meters.left_mse_loss_coord_train_for_compare[self.temps.epoch])
        right_mse_loss_coord_train_for_compare = np.mean(
            self.meters.right_mse_loss_coord_train_for_compare[self.temps.epoch])
        total_pred_mse_loss_coord_train_for_compare = np.mean(
            self.meters.total_pred_mse_loss_coord_train_for_compare[self.temps.epoch])
        left_mse_loss_coord_val_for_compare = np.mean(
            self.meters.left_mse_loss_coord_val_for_compare[self.temps.epoch])
        right_mse_loss_coord_val_for_compare = np.mean(
            self.meters.right_mse_loss_coord_val_for_compare[self.temps.epoch])
        total_pred_mse_loss_coord_val_for_compare = np.mean(
            self.meters.total_pred_mse_loss_coord_val_for_compare[self.temps.epoch])
        left_prec_coord_train = np.mean(
            self.meters.left_prec_coord_train[self.temps.epoch])
        right_prec_coord_train = np.mean(
            self.meters.right_prec_coord_train[self.temps.epoch])
        total_pred_prec_coord_train = np.mean(
            self.meters.total_pred_prec_coord_train[self.temps.epoch])
        left_prec_coord_val = np.mean(
            self.meters.left_prec_coord_val[self.temps.epoch])
        right_prec_coord_val = np.mean(
            self.meters.right_prec_coord_val[self.temps.epoch])
        total_pred_prec_coord_val = np.mean(
            self.meters.total_pred_prec_coord_val[self.temps.epoch])
        loss_coord_train = np.mean(
            self.meters.loss_coord_train[self.temps.epoch])
        loss_coord_val = np.mean(self.meters.loss_coord_val[self.temps.epoch])

        with open(self.prec_loss_val_record_file, 'a') as outfile:
            log = f"epoch: {epoch}, total_pred_prec_coord_train: {total_pred_prec_coord_train:.4f}, total_pred_prec_coord_val: {total_pred_prec_coord_val:.4f},\
            left_prec_coord_train: {left_prec_coord_train:.4f}, left_prec_coord_val: {left_prec_coord_val:.4f},\
            right_prec_coord_train: {right_prec_coord_train:.4f}, right_prec_coord_val: {right_prec_coord_val:.4f},\
             loss_coord_train: {loss_coord_train:.4f}, \
            loss_coord_val: {loss_coord_val:.4f}, \
            total_pred_mse_loss_coord_train_for_compare: {total_pred_mse_loss_coord_train_for_compare:.4f}, \
            left_mse_loss_coord_train_for_compare: {left_mse_loss_coord_train_for_compare:.4f}, \
            right_mse_loss_coord_train_for_compare: {right_mse_loss_coord_train_for_compare:.4f}, \
            total_pred_mse_loss_coord_val_for_compare: {total_pred_mse_loss_coord_val_for_compare:.4f},\
            left_mse_loss_coord_val_for_compare: {left_mse_loss_coord_val_for_compare:.4f},\
            right_mse_loss_coord_val_for_compare: {right_mse_loss_coord_val_for_compare:.4f}  "

            outfile.write(log + "\n")
        infofmt = "*[{temps.epoch}]\t" \
                  "total_pred_prec_coord_train: {total_pred_prec_coord_train:.4f} total_pred_prec_coord_val: {total_pred_prec_coord_val:.4f}\t" \
                  "left_prec_coord_train: {left_prec_coord_train:.4f} left_prec_coord_val: {left_prec_coord_val:.4f}\t" \
                  "right_prec_coord_train: {right_prec_coord_train:.4f} right_prec_coord_val: {right_prec_coord_val:.4f}\t" \
                  "loss_coord_train: {loss_coord_train:.4f} loss_coord_val: {loss_coord_val:.4f}\t" \
                  "total_pred_mse_loss_coord_train_for_compare: {total_pred_mse_loss_coord_train_for_compare:.4f}\t" \
                  "left_mse_loss_coord_train_for_compare: {left_mse_loss_coord_train_for_compare:.4f}\t" \
                  "right_mse_loss_coord_train_for_compare: {right_mse_loss_coord_train_for_compare:.4f}\t" \
                  "total_pred_mse_loss_coord_val_for_compare: {total_pred_mse_loss_coord_val_for_compare:.4f}\t"\
                  "left_mse_loss_coord_val_for_compare: {left_mse_loss_coord_val_for_compare:.4f}\t"\
                  "right_mse_loss_coord_val_for_compare: {right_mse_loss_coord_val_for_compare:.4f}\t"

        infodict = dict(
            temps=self.temps,
            total_pred_prec_coord_train=np.mean(
                self.meters.total_pred_prec_coord_train[self.temps.epoch]),
            left_prec_coord_train=np.mean(
                self.meters.left_prec_coord_train[self.temps.epoch]),
            right_prec_coord_train=np.mean(
                self.meters.right_prec_coord_train[self.temps.epoch]),
            total_pred_prec_coord_val=np.mean(
                self.meters.total_pred_prec_coord_val[self.temps.epoch]),
            left_prec_coord_val=np.mean(
                self.meters.left_prec_coord_val[self.temps.epoch]),
            right_prec_coord_val=np.mean(
                self.meters.right_prec_coord_val[self.temps.epoch]),
            loss_coord_train=np.mean(
                self.meters.loss_coord_train[self.temps.epoch]),
            loss_coord_val=np.mean(
                self.meters.loss_coord_val[self.temps.epoch]),
            total_pred_mse_loss_coord_train_for_compare=np.mean(
                self.meters.total_pred_mse_loss_coord_train_for_compare[self.temps.epoch]),
            left_mse_loss_coord_train_for_compare=np.mean(
                self.meters.left_mse_loss_coord_train_for_compare[self.temps.epoch]),
            right_mse_loss_coord_train_for_compare=np.mean(
                self.meters.right_mse_loss_coord_train_for_compare[self.temps.epoch]),
            total_pred_mse_loss_coord_val_for_compare=np.mean(
                self.meters.total_pred_mse_loss_coord_val_for_compare[self.temps.epoch]),
            left_mse_loss_coord_val_for_compare=np.mean(
                self.meters.left_mse_loss_coord_val_for_compare[self.temps.epoch]),
            right_mse_loss_coord_val_for_compare=np.mean(
                self.meters.right_mse_loss_coord_val_for_compare[self.temps.epoch])
        )
        self.total_pred_train_angular_precison_record[epoch] = total_pred_prec_coord_train
        self.left_train_angular_precison_record[epoch] = left_prec_coord_train
        self.right_train_angular_precison_record[epoch] = right_prec_coord_train
        self.total_pred_val_angular_precison_record[epoch] = total_pred_prec_coord_val
        self.left_val_angular_precison_record[epoch] = left_prec_coord_val
        self.right_val_angular_precison_record[epoch] = right_prec_coord_val
        self.train_loss_yaw_pitch_record[epoch] = loss_coord_train
        self.val_loss_yaw_pitch_record[epoch] = loss_coord_val
        self.total_pred_mse_loss_coord_train_for_compare_record[
            epoch] = total_pred_mse_loss_coord_train_for_compare
        self.left_mse_loss_coord_train_for_compare_record[epoch] = left_mse_loss_coord_train_for_compare
        self.right_mse_loss_coord_train_for_compare_record[
            epoch] = right_mse_loss_coord_train_for_compare
        self.total_pred_mse_loss_coord_val_for_compare_record[
            epoch] = total_pred_mse_loss_coord_val_for_compare
        self.left_mse_loss_coord_val_for_compare_record[epoch] = left_mse_loss_coord_val_for_compare
        self.right_mse_loss_coord_val_for_compare_record[epoch] = right_mse_loss_coord_val_for_compare
        self.temps.base_logger.info(infofmt.format(**infodict))

    def _log_headpose(self):
        infofmt = "*[{temps.epoch}]\t" \
                  "loss_depth_train: {loss_depth_train:.4f} loss_depth_val: {loss_depth_val:.4f}\t"
        infodict = dict(
            temps=self.temps,
            loss_depth_train=np.mean(
                self.meters.loss_depth_train[self.temps.epoch]),
            loss_depth_val=np.mean(
                self.meters.loss_depth_val[self.temps.epoch]),
        )
        self.temps.headpose_logger.info(infofmt.format(**infodict))

    def _train_base_epoch(self):
        logging.info(
            f'=====================lr: {self.temps.lr}===================')
        logger = self.temps.base_logger.getChild('epoch')
        # prepare models
        resnet = self._prepare_model(self.models.resnet)
        decoder = self._prepare_model(self.models.decoder)
        # refine_depth = self._prepare_model(self.models.refine_depth, train=True)
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")

        resnet_total = sum([param.nelement() for param in resnet.parameters()])
        decoder_total = sum([param.nelement()
                            for param in decoder.parameters()])
        # print("Number of parameters in resnet_total: %.2fM" % (resnet_total/1e6))
        # print("Number of parameters in decoder_total: %.2fM" % (decoder_total/1e6))
        # total = resnet_total + decoder_total
        # print("Number of parameters in three_part parameter: %.2fM" % (total/1e6))

        AWingloss = self.adaptive_wing_loss
        L1_loss = th.nn.L1Loss().cuda()
        # prepare solvers
        if self.temps.fine_tune_headpose:

            # self.temps.base_solver = optim.Adam(self._group_weight(self.models.resnet, lr=self.temps.lr) +
            #                                    self._group_weight(self.models.decoder, lr=self.temps.lr) +
            #                                    self._group_weight(self.models.refine_depth, lr=self.temps.lr),
            #                                    weight_decay=5e-4)

            self.temps.base_solver = optim.SGD(self._group_weight(self.models.resnet, lr=self.temps.lr) +
                                               self._group_weight(
                                                   self.models.decoder, lr=self.temps.lr),  # +
                                               #self._group_weight(self.models.refine_depth, lr=self.temps.lr),
                                               weight_decay=5e-4)
            # self.temps.base_solver = optim.SGD(list(resnet.parameters()) +
            #                                    list(decoder.parameters()) +
            #                                    list(refine_depth.parameters()),
            #                                    lr=self.temps.lr, weight_decay=5e-4)
        else:
            self.temps.base_solver = optim.SGD(self._group_weight(self.models.resnet, lr=self.temps.lr) +
                                               self._group_weight(
                                                   self.models.decoder, lr=self.temps.lr),
                                               weight_decay=5e-4)
        self.timeit()
        for i,  (data, label) in enumerate(self.temps.train_loader):
            self.temps.iter = i
            # prepare data
            # face_image = data["face_rgb"].to(device)
            # face_depth = data["face_depth"].to(device)
            # r= depth[0]
            # g= depth[1]
            # b= depth[2]
            # print("r==g",r==g)
            # import pudb
            # pudb.set_trace()
            left_eye_image = data["left_rgb"].to(device)
            right_eye_image = data["right_rgb"].to(device)
            gaze_gt = label.to(device)

            # print("face_image.shape",face_image.shape)
            # print("face_depth.shape",face_image.shape)

            # measure data loading time
            self.temps.data_time = self._timeit()

            # forward
            lfeat = resnet(left_eye_image)
            rfeat = resnet(right_eye_image)


            left_gaze_pred, right_gaze_pred, total_pred = decoder(
                lfeat, rfeat)  # , head_pose)
            left_right_gaze_concat = th.cat(
                [th.cat([left_gaze_pred, right_gaze_pred], 1), total_pred], 1)
            left_mse_loss_coord = F.mse_loss(left_gaze_pred, gaze_gt)
            right_mse_loss_coord = F.mse_loss(right_gaze_pred, gaze_gt)
            total_pred_mse_loss_coord = F.mse_loss(total_pred, gaze_gt)
            loss_coord = L1_loss(left_right_gaze_concat, th.cat(
                [th.cat([gaze_gt, gaze_gt], 1), gaze_gt], 1))  # AWingloss(gaze_pred, gaze_gt)

            # update resnet & decoder
            self.temps.base_solver.zero_grad()
            loss_coord.backward()
            self.temps.base_solver.step()

            # record loss & accuracy
            left_accs_iter = 0
            left_count_iter = 0
            right_accs_iter = 0
            right_count_iter = 0
            total_pred_accs_iter = 0
            total_pred_count_iter = 0

            for k, gaze in enumerate(left_gaze_pred):
                gaze = gaze.cpu().detach().numpy()
                left_count_iter += 1
                left_accs_iter += self._precision(gazeto3d(gaze),
                                                  gazeto3d(gaze_gt.cpu().numpy()[k]))

            for k, gaze in enumerate(right_gaze_pred):
                gaze = gaze.cpu().detach().numpy()
                right_count_iter += 1
                right_accs_iter += self._precision(
                    gazeto3d(gaze), gazeto3d(gaze_gt.cpu().numpy()[k]))

            for k, gaze in enumerate(total_pred):
                gaze = gaze.cpu().detach().numpy()
                total_pred_count_iter += 1
                total_pred_accs_iter += self._precision(
                    gazeto3d(gaze), gazeto3d(gaze_gt.cpu().numpy()[k]))

            left_prec_coord_iter = left_accs_iter/left_count_iter
            right_prec_coord_iter = right_accs_iter/right_count_iter
            total_pred_prec_coord_iter = total_pred_accs_iter/total_pred_count_iter
            epoch = self.temps.epoch
            self.meters.loss_coord_train[epoch].append(loss_coord.item())
            self.meters.left_prec_coord_train[epoch].append(
                left_prec_coord_iter.item())
            self.meters.right_prec_coord_train[epoch].append(
                right_prec_coord_iter.item())
            self.meters.total_pred_prec_coord_train[epoch].append(
                total_pred_prec_coord_iter.item())
            self.meters.left_mse_loss_coord_train_for_compare[epoch].append(
                left_mse_loss_coord.item())
            self.meters.right_mse_loss_coord_train_for_compare[epoch].append(
                right_mse_loss_coord.item())
            self.meters.total_pred_mse_loss_coord_train_for_compare[epoch].append(
                total_pred_mse_loss_coord.item())

            # measure batch time
            self.temps.batch_time = self._timeit()

            # logging
            info = f"[{self.temps.epoch}][{self.temps.iter}/{self.temps.num_iters}]\t" \
                   f"data_time: {self.temps.data_time:.2f} batch_time: {self.temps.batch_time:.2f}\t" \
                   f"left_prec_coord_train: {self.meters.left_prec_coord_train[self.temps.epoch][-1]:.4f}\t" \
                   f"right_prec_coord_train: {self.meters.right_prec_coord_train[self.temps.epoch][-1]:.4f}\t" \
                   f"AWing_loss_coord_train: {self.meters.loss_coord_train[self.temps.epoch][-1]:.4f}\t"\
                   f"left_mse_loss_coord_for_compare: {self.meters.left_mse_loss_coord_train_for_compare[self.temps.epoch][-1]:.4f}\t"\
                   f"right_mse_loss_coord_for_compare: {self.meters.right_mse_loss_coord_train_for_compare[self.temps.epoch][-1]:.4f}\t"

            # infodict = dict(
            #     temps=self.temps,
            #     prec_coord_train=self.meters.prec_coord_train[self.temps.epoch][-1],
            #     loss_coord_train=self.meters.loss_coord_train[self.temps.epoch][-1],
            # )
            logger.info(info)

    def _train_headpose_epoch(self):
        logger = self.temps.headpose_logger.getChild('epoch')
        # prepare models
        refine_depth = self._prepare_model(self.models.refine_depth)
        depth_loss = self.models.depth_loss
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")
        # prepare solvers
        self.temps.headpose_solver = optim.SGD(self._group_weight(self.models.refine_depth, lr=self.temps.lr),
                                               weight_decay=5e-4)

        self.timeit()
        for i, batch in enumerate(self.temps.train_loader):
            self.temps.iter = i
            # prepare data
            face_image = batch['face_image'].to(device)
            face_depth = batch['face_depth'].to(device)

            # measure data loading time
            self.temps.data_time = self._timeit()

            # forward
            head_pose, refined_depth = refine_depth(face_image, face_depth)
            loss_depth = depth_loss(refined_depth, face_depth)

            # update resnet & decoder
            self.temps.headpose_solver.zero_grad()
            loss_depth.backward()
            self.temps.headpose_solver.step()

            # record loss & accuracy
            epoch = self.temps.epoch
            self.meters.loss_depth_train[epoch].append(loss_depth.item())

            # visualize and save results
            face_depth.detach_()
            refined_depth.detach_()
            # with th.no_grad():
            #     depth_grid_gt = make_grid(face_depth).cpu()
            #     depth_grid_rf = make_grid(refined_depth).cpu()
            #     with vplt.set_draw(name='train_depth_groundtruth') as ax:
            #         ax.imshow(depth_grid_gt.numpy().transpose((1, 2, 0)))
            #     with vplt.set_draw(name='train_depth_refined') as ax:
            #         ax.imshow(depth_grid_rf.numpy().transpose((1, 2, 0)))
            #     save_image(depth_grid_gt, os.path.join(self.result_root, "train", "depth", f"ep{self.temps.epoch:02d}iter{i:04d}_gt.png"))
            #     save_image(depth_grid_rf, os.path.join(self.result_root, "train", "depth", f"ep{self.temps.epoch:02d}iter{i:04d}_rf.png"))

            # measure batch time
            self.temps.batch_time = self._timeit()

            # logging
            infofmt = "[{temps.epoch}][{temps.iter}/{temps.num_iters}]\t" \
                      "data_time: {temps.data_time:.2f} batch_time: {temps.batch_time:.2f}\t" \
                      "loss_depth_train: {loss_depth_train:.4f} "
            infodict = dict(
                temps=self.temps,
                loss_depth_train=self.meters.loss_depth_train[epoch][-1],
            )
            logger.info(infofmt.format(**infodict))

    def _test_base(self):
        logger = self.temps.base_logger.getChild('val')
        resnet = self._prepare_model(self.models.resnet, train=False)
        decoder = self._prepare_model(self.models.decoder, train=False)
        AWingloss = self.adaptive_wing_loss
        L1_loss = th.nn.L1Loss().cuda()
        #refine_depth = self._prepare_model(self.models.refine_depth, train=True)
        loss_lcoord, loss_rcoord, loss_coord, prec_lcoord, prec_rcoord, left_prec_coord, right_prec_coord, num_batches = 0, 0, 0, 0, 0, 0, 0, 0
        total_pred_prec_coord = 0
        left_mse_coord_for_compare = 0
        right_mse_coord_for_compare = 0
        total_pred_mse_coord_for_compare = 0
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")

        for i, (data, label) in enumerate(self.temps.val_loader):
            self.temps.iter = i
            # prepare data
            # face_image = data["face_rgb"].to(device)
            # face_depth = data["face_depth"].to(device)
            left_eye_image = data["left_rgb"].to(device)
            right_eye_image = data["right_rgb"].to(device)
            gaze_gt = label.to(device)

            # forward
            with th.no_grad():
                lfeat = resnet(left_eye_image)
                rfeat = resnet(right_eye_image)
                #head_pose, refined_depth = refine_depth(face_image, face_depth)
                left_gaze_pred, right_gaze_pred, total_pred = decoder(
                    lfeat, rfeat)  # , head_pose)
                left_mse_loss_coord_iter = F.mse_loss(left_gaze_pred, gaze_gt)
                right_mse_loss_coord_iter = F.mse_loss(
                    right_gaze_pred, gaze_gt)
                total_pred_mse_loss_coord_iter = F.mse_loss(
                    total_pred, gaze_gt)
                left_right_gaze_concat = th.cat(
                    [th.cat([left_gaze_pred, right_gaze_pred], 1), total_pred], 1)
                loss_coord_iter = L1_loss(left_right_gaze_concat, th.cat(
                    [th.cat([gaze_gt, gaze_gt], 1), gaze_gt], 1))
                #loss_coord_iter  = AWingloss(gaze_pred, gaze_gt)

                left_accs_iter = 0
                left_count_iter = 0
                right_accs_iter = 0
                right_count_iter = 0
                total_pred_accs_iter = 0
                total_pred_count_iter = 0

                for k, gaze in enumerate(total_pred):
                    gaze = gaze.cpu().detach().numpy()
                    total_pred_count_iter += 1
                    total_pred_accs_iter += self._precision(
                        gazeto3d(gaze), gazeto3d(gaze_gt.cpu().numpy()[k]))

                for k, gaze in enumerate(left_gaze_pred):
                    gaze = gaze.cpu().detach().numpy()
                    left_count_iter += 1
                    left_accs_iter += self._precision(
                        gazeto3d(gaze), gazeto3d(gaze_gt.cpu().numpy()[k]))

                for k, gaze in enumerate(right_gaze_pred):
                    gaze = gaze.cpu().detach().numpy()
                    right_count_iter += 1
                    right_accs_iter += self._precision(
                        gazeto3d(gaze), gazeto3d(gaze_gt.cpu().numpy()[k]))

                left_prec_coord_iter = left_accs_iter/left_count_iter
                right_prec_coord_iter = right_accs_iter/right_count_iter
                total_pred_prec_coord_iter = total_pred_accs_iter/total_pred_count_iter

            # accumulate meters
            loss_coord += loss_coord_iter.item()
            left_prec_coord += left_prec_coord_iter.item()
            right_prec_coord += right_prec_coord_iter.item()
            total_pred_prec_coord += total_pred_prec_coord_iter.item()
            left_mse_coord_for_compare += left_mse_loss_coord_iter.item()
            right_mse_coord_for_compare += right_mse_loss_coord_iter.item()
            total_pred_mse_coord_for_compare += total_pred_mse_loss_coord_iter.item()
            num_batches += 1
            # logging
            infofmt = "[{temps.epoch}]\t" \
                      "left_prec_coord: {left_prec_coord: .4f}\t" \
                      "right_prec_coord: {right_prec_coord: .4f}\t" \
                      "total_pred_prec_coord: {total_pred_prec_coord: .4f}\t" \
                      "AWing_loss_coord: {loss_coord: .4f}\t"\
                      "left_mse_loss_coord for compare:{left_mse_loss_coord: .4f}\t"\
                      "right_mse_loss_coord for compare:{right_mse_loss_coord: .4f}\t" \
                      "total_pred_loss_coord for compare:{total_pred_mse_loss_coord: .4f}"
            infodict = dict(
                temps=self.temps,
                left_prec_coord=left_prec_coord_iter,
                right_prec_coord=right_prec_coord_iter,
                total_pred_prec_coord=total_pred_prec_coord_iter,
                loss_coord=loss_coord_iter,
                left_mse_loss_coord=left_mse_loss_coord_iter,
                right_mse_loss_coord=right_mse_loss_coord_iter,
                total_pred_mse_loss_coord=total_pred_mse_loss_coord_iter
            )
            logger.info(infofmt.format(**infodict))

        # record meters
        epoch = self.temps.epoch
        self.meters.loss_coord_val[epoch] = loss_coord / num_batches
        self.meters.left_prec_coord_val[epoch] = left_prec_coord / num_batches
        self.meters.right_prec_coord_val[epoch] = right_prec_coord / num_batches
        self.meters.total_pred_prec_coord_val[epoch] = total_pred_prec_coord / num_batches
        self.meters.left_mse_loss_coord_val_for_compare[epoch] = left_mse_coord_for_compare / num_batches
        self.meters.right_mse_loss_coord_val_for_compare[
            epoch] = right_mse_coord_for_compare / num_batches
        self.meters.total_pred_mse_loss_coord_val_for_compare[
            epoch] = total_pred_mse_coord_for_compare / num_batches

    def _test_headpose(self):
        logger = self.temps.headpose_logger.getChild('val')
        refine_depth = self._prepare_model(self.models.refine_depth)
        depth_loss = self.models.depth_loss
        loss_depth, num_batchs = 0, 0
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")
        for i, batch in enumerate(self.temps.val_loader):
            self.temps.iter = i
            # prepare data
            face_image = batch['face_image'].to(device)
            face_depth = batch['face_depth'].to(device)

            # measure data loading time
            self.temps.data_time = self._timeit()

            # forward
            with th.no_grad():
                head_pose, refined_depth = refine_depth(face_image, face_depth)
                loss_depth_iter = depth_loss(refined_depth, face_depth)
                depth_grid_gt = make_grid(face_depth).cpu()
                depth_grid_rf = make_grid(refined_depth).cpu()
                with vplt.set_draw(name='val_depth_groundtruth') as ax:
                    ax.imshow(depth_grid_gt.numpy().transpose((1, 2, 0)))
                with vplt.set_draw(name='val_train_depth_refined') as ax:
                    ax.imshow(depth_grid_rf.numpy().transpose((1, 2, 0)))
                save_image(depth_grid_gt, os.path.join(self.result_root, "val", "depth",
                                                       f"ep{self.temps.epoch:02d}iter{i:04d}_gt.png"))
                save_image(depth_grid_rf, os.path.join(self.result_root, "val", "depth",
                                                       f"ep{self.temps.epoch:02d}iter{i:04d}_rf.png"))

            # accumulate meters
            loss_depth += loss_depth_iter.item()
            num_batchs += 1
            # logging
            infofmt = "[{temps.epoch}]\t" \
                      "loss_depth: {loss_depth:.4f}"
            infodict = dict(
                temps=self.temps,
                loss_depth=loss_depth_iter,
            )
            logger.info(infofmt.format(**infodict))

        # record meters
        epoch = self.temps.epoch
        self.meters.loss_depth_val[epoch] = loss_depth / num_batchs

    def angular(gaze, label):
        total = np.sum(gaze * label)
        return np.arccos(min(total/(np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999))*180/np.pi

    @staticmethod
    def _precision(gaze, label):
        total = np.sum(gaze * label)
        return np.arccos(min(total/(np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999))*180/np.pi
        # return th.mean(th.sqrt(th.sum((out - target) ** 2, 1)))


if __name__ == '__main__':

    #logging.basicConfig(level=logging.INFO, format='<%(name)s:%(levelname)s> %(message)s')

    testindex_fold = 1
    result_save = '/home/workspace/your_save_path/SaveResult'
    model_name = "gaze_eyediap_v8_depth3_resnet_SEPRE_lookat_resnet_correct_left_right_feature_branch"

    experiment_name = model_name
    result_save_log_path = os.path.join(result_save,
                                        time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())) + experiment_name+"_person" + str(testindex_fold) + "_testindex"+"_batchsize4")

    if not os.path.exists(result_save_log_path):
        os.makedirs(result_save_log_path)

    log_file = os.path.join(result_save_log_path, experiment_name + "_person" +
                            str(testindex_fold) + "_testindex" + "_batchsize4" + '_exp.log')

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s',
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    trainer = GazeTrainer(
        batch_size_train=4,
        batch_size_val=4,
        num_workers=2,
        # trainer parameters
        is_cuda=True,
        exp_name=model_name,
        result_root=result_save_log_path,
        test_choose=testindex_fold
    )
    #trainer.train_headpose(10, lr=2e-1)
    learning_rate = 1e-5
    total_epochs = 30

    # ============================================ Training ===========================================
    logging.info('============ Train stage ============')
    logging.info(f'===========total epochs:{total_epochs}============')
    logging.info("learning rate setting: \nif(epoch<20):\nself.temps.lr =  1e-2\n elif(epoch<30):\nself.temps.lr =  1e-3\nelse:\nself.temps.lr =  1e-4")
    #logging.info("learning rate setting: \nif(epoch<10):\nself.temps.lr =  1e-2\n elif(epoch<15):\nself.temps.lr =  1e-3\nelse:\nself.temps.lr =  1e-4")
    #logging.info("learning rate setting: \nif(epoch<40):\nself.temps.lr =  1e-2\n elif(epoch<60):\nself.temps.lr =  1e-3\nelse:\nself.temps.lr =  1e-4")
    #   def train_base(self, epochs, lr=1e-4, use_refined_depth=False, fine_tune_headpose=True):
    trainer.train_base(epochs=total_epochs, lr=learning_rate,
                       use_refined_depth=True)
    logging.info('============ congratulations!done ============')
    left_train_angular_precison_record = trainer.left_train_angular_precison_record
    right_train_angular_precison_record = trainer.right_train_angular_precison_record
    total_pred_train_angular_precison_record = trainer.total_pred_train_angular_precison_record

    left_val_angular_precison_record = trainer.left_val_angular_precison_record
    right_val_angular_precison_record = trainer.right_val_angular_precison_record
    total_pred_val_angular_precison_record = trainer.total_pred_val_angular_precison_record

    train_loss_yaw_pitch_record = trainer.train_loss_yaw_pitch_record
    val_loss_yaw_pitch_record = trainer.val_loss_yaw_pitch_record

    left_mse_coord_train_record = trainer.left_mse_loss_coord_train_for_compare_record
    right_mse_coord_train_record = trainer.right_mse_loss_coord_train_for_compare_record
    total_pred_mse_coord_train_record = trainer.total_pred_mse_loss_coord_train_for_compare_record

    left_mse_coord_val_record = trainer.left_mse_loss_coord_val_for_compare_record
    right_mse_coord_val_record = trainer.right_mse_loss_coord_val_for_compare_record
    total_pred_mse_coord_val_record = trainer.total_pred_mse_loss_coord_val_for_compare_record

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'left_train_angular_precison_record:')
    logging.info(left_train_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'right_train_angular_precison_record:')
    logging.info(right_train_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'total_pred_train_angular_precison_record:')
    logging.info(total_pred_train_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'left_val_angular_precison_record:')
    logging.info(left_val_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'right_val_angular_precison_record:')
    logging.info(right_val_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'total_pred_val_angular_precison_record:')
    logging.info(total_pred_val_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'train_loss_yaw_pitch_record:')
    logging.info(train_loss_yaw_pitch_record)

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'val_loss_yaw_pitch_record:')
    logging.info(val_loss_yaw_pitch_record)

    logging.info('=====================sorted===================')

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'left_sorted train_angular_precison_record:')
    left_sorted_train_angular_precison_record = sorted(
        left_train_angular_precison_record.items(), key=lambda x: x[1])
    logging.info(left_sorted_train_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'right sorted train_angular_precison_record:')
    right_sorted_train_angular_precison_record = sorted(
        right_train_angular_precison_record.items(), key=lambda x: x[1])
    logging.info(right_sorted_train_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    logging.info(f'total_pred sorted train_angular_precison_record:')
    total_pred_sorted_train_angular_precison_record = sorted(
        total_pred_train_angular_precison_record.items(), key=lambda x: x[1])
    logging.info(total_pred_sorted_train_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    left_sorted_val_angular_precison_record = sorted(
        left_val_angular_precison_record.items(), key=lambda x: x[1])
    logging.info(f'left sorted val_angular_precison_record:')
    logging.info(left_sorted_val_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')
    right_sorted_val_angular_precison_record = sorted(
        right_val_angular_precison_record.items(), key=lambda x: x[1])
    logging.info(f'right sorted val_angular_precison_record:')
    logging.info(right_sorted_val_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    total_pred_sorted_val_angular_precison_record = sorted(
        total_pred_val_angular_precison_record.items(), key=lambda x: x[1])
    logging.info(f'sorted total_pred val_angular_precison_record:')
    logging.info(total_pred_sorted_val_angular_precison_record)

    logging.info(f'\n')
    logging.info(f'\n')

    sorted_train_loss_yaw_pitch_record = sorted(
        train_loss_yaw_pitch_record.items(), key=lambda x: x[1])
    logging.info(f'sorted train_AWING_loss_yaw_pitch_record:')
    logging.info(sorted_train_loss_yaw_pitch_record)

    logging.info(f'\n')
    logging.info(f'\n')

    sorted_val_loss_yaw_pitch_record = sorted(
        val_loss_yaw_pitch_record.items(), key=lambda x: x[1])
    logging.info(f'sorted val_AWING_loss_yaw_pitch_record:')
    logging.info(sorted_val_loss_yaw_pitch_record)

    left_sorted_mse_coord_train_record = sorted(
        left_mse_coord_train_record.items(), key=lambda x: x[1])
    logging.info(f'sorted left train_MSE_yaw_pitch_record:')
    logging.info(left_sorted_mse_coord_train_record)

    right_sorted_mse_coord_train_record = sorted(
        right_mse_coord_train_record.items(), key=lambda x: x[1])
    logging.info(f'sorted right  train_MSE_yaw_pitch_record:')
    logging.info(right_sorted_mse_coord_train_record)

    total_pred_sorted_mse_coord_train_record = sorted(
        total_pred_mse_coord_train_record.items(), key=lambda x: x[1])
    logging.info(f'sorted total_pred  train_MSE_yaw_pitch_record:')
    logging.info(total_pred_sorted_mse_coord_train_record)

    left_sorted_mse_coord_val_record = sorted(
        left_mse_coord_val_record.items(), key=lambda x: x[1])
    logging.info(f'sorted left val_MSE_yaw_pitch_record:')
    logging.info(left_sorted_mse_coord_val_record)

    right_sorted_mse_coord_val_record = sorted(
        right_mse_coord_val_record.items(), key=lambda x: x[1])
    logging.info(f'sorted right  val_MSE_yaw_pitch_record:')
    logging.info(right_sorted_mse_coord_val_record)

    total_pred_sorted_mse_coord_val_record = sorted(
        total_pred_mse_coord_val_record.items(), key=lambda x: x[1])
    logging.info(f'sorted total_pred  val_MSE_yaw_pitch_record:')
    logging.info(total_pred_sorted_mse_coord_val_record)

    metric_log = trainer.prec_loss_val_record_file
    with open(metric_log, 'a') as outfile:
        outfile.write("\n\n================================\n")
        outfile.write("\nsorted train_loss_yaw_pitch_record:" + "\n")
        outfile.write(json.dumps(sorted_train_loss_yaw_pitch_record))
        outfile.write("\nsorted val_loss_yaw_pitch_record:" + "\n")
        outfile.write(json.dumps(sorted_val_loss_yaw_pitch_record))
        outfile.write("\n\n================================\n")

        outfile.write("left_sorted train_angular_precison_record:" + "\n")
        outfile.write(json.dumps(left_sorted_train_angular_precison_record))
        outfile.write("\nleft_sorted val_angular_precison_record:" + "\n")
        outfile.write(json.dumps(left_sorted_val_angular_precison_record))

        outfile.write(
            "\nleft sorted train_MSE_yaw_pitch_record_for_compare:" + "\n")
        outfile.write(json.dumps(left_sorted_mse_coord_train_record))
        outfile.write(
            "\nleft sorted val_MSE_yaw_pitch_record_for_compare:" + "\n")
        outfile.write(json.dumps(left_sorted_mse_coord_val_record))

        outfile.write("\n\n================================\n")

        outfile.write("right sorted train_angular_precison_record:" + "\n")
        outfile.write(json.dumps(right_sorted_train_angular_precison_record))
        outfile.write("\nright sorted val_angular_precison_record:" + "\n")
        outfile.write(json.dumps(right_sorted_val_angular_precison_record))

        outfile.write(
            "\nright sorted train_MSE_yaw_pitch_record_for_compare:" + "\n")
        outfile.write(json.dumps(right_sorted_mse_coord_train_record))
        outfile.write(
            "\nright sorted val_MSE_yaw_pitch_record_for_compare:" + "\n")
        outfile.write(json.dumps(right_sorted_mse_coord_val_record))

        outfile.write(
            "\ntotal_pred sorted train_MSE_yaw_pitch_record_for_compare:" + "\n")
        outfile.write(json.dumps(total_pred_sorted_mse_coord_train_record))
        outfile.write(
            "\ntotal_pred sorted val_MSE_yaw_pitch_record_for_compare:" + "\n")
        outfile.write(json.dumps(total_pred_sorted_mse_coord_val_record))
        outfile.write(
            "\ntotal_pred sorted train_angular_precison_record:" + "\n")
        outfile.write(json.dumps(
            total_pred_sorted_train_angular_precison_record))
        outfile.write(
            "\ntotal_pred sorted val_angular_precison_record:" + "\n")
        outfile.write(json.dumps(
            total_pred_sorted_val_angular_precison_record))
