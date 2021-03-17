import os
import cv2
import random
import numpy as np
from PIL import Image
from distutils.version import LooseVersion
from time import time

import torch
import torch.nn.functional as F
import torchvision.transforms as tf

from models.baseline_same import Baseline as UNet
from utils.disp import tensor_to_image
from utils.disp import colors_256 as colors
from bin_mean_shift import Bin_Mean_Shift
from modules import get_coordinate_map
from utils.loss import Q_loss
from instance_parameter_loss import InstanceParameterLoss

class Cfg():
    def __init__(self):
        self.arch = 'resnet101'
        self.pretrained = False
        self.embed_dims = 2
        self.fix_bn = False

class Predictor():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = Cfg()
        # build network
        self.network = UNet(self.cfg)

        self.model_dict = torch.load('pretrained.pt', map_location=lambda storage, loc: storage)
        self.network.load_state_dict(self.model_dict)

        self.network.to(self.device)
        self.network.eval()

        self.transforms = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.bin_mean_shift = Bin_Mean_Shift(device=self.device)
        self.k_inv_dot_xy1 = get_coordinate_map(self.device)
        self.instance_parameter_loss = InstanceParameterLoss(self.k_inv_dot_xy1)

        self.h, self.w = 192, 256
    
    def predict(self,image_path):
        with torch.no_grad():
            image = cv2.imread(image_path)
            # the network is trained with 192*256 and the intrinsic parameter is set as ScanNet
            image = cv2.resize(image, (self.w, self.h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.transforms(image)
            image = image.to(self.device).unsqueeze(0)
            
            # forward pass
            logit, embedding, _, _, param = self.network(image)
            
            prob = torch.sigmoid(logit[0])
            
            # infer per pixel depth using per pixel plane parameter, currently Q_loss need a dummy gt_depth as input
            _, _, per_pixel_depth = Q_loss(param, self.k_inv_dot_xy1, torch.ones_like(logit))

            # fast mean shift
            segmentation, sampled_segmentation, sample_param = self.bin_mean_shift.test_forward(
                prob, embedding[0], param, mask_threshold=0.1)

            # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned, 
            # we thus use avg_pool_2d to smooth the segmentation results
            b = segmentation.t().view(1, -1, self.h, self.w)
            pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
            b = pooling_b.view(-1, self.h*self.w).t()
            segmentation = b

            # infer instance depth
            instance_loss, instance_depth, instance_abs_disntace, instance_parameter = self.instance_parameter_loss(
                segmentation, sampled_segmentation, sample_param, torch.ones_like(logit), torch.ones_like(logit), False)

            # return cluster results
            predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

            # mask out non planar region
            predict_segmentation[prob.cpu().numpy().reshape(-1) <= 0.1] = 20
            predict_segmentation = predict_segmentation.reshape(self.h, self.w)

            # visualization and evaluation
            image = tensor_to_image(image.cpu()[0])
            mask = (prob > 0.1).float().cpu().numpy().reshape(self.h, self.w)
            depth = instance_depth.cpu().numpy()[0, 0].reshape(self.h, self.w)
            per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(self.h, self.w)

            # use per pixel depth for non planar region
            depth = depth * (predict_segmentation != 20) + per_pixel_depth * (predict_segmentation == 20)

            # change non planar to zero, so non planar region use the black color
            predict_segmentation += 1
            predict_segmentation[predict_segmentation == 21] = 0

            return depth,predict_segmentation,mask,instance_parameter

