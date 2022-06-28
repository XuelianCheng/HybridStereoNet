import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from models.leastereo.skip_model_3d import newMatching
from models.leastereo.decoding_formulas import network_layer_to_space
from models.feature_net import FeatureNet
from time import time

import pdb
import copy
import logging

class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.maxdisp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.maxdisp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

class Disp(nn.Module):
    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        # x = F.interpolate(x, [self.maxdisp, x.size()[3]*4, x.size()[4]*4], mode='trilinear', align_corners=False)
        x = F.interpolate(x, [self.maxdisp, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)      
        x = self.disparity(x)
        return x

class HybridStereoNet(nn.Module):
    def __init__(self, args, zero_head=False):
        super(HybridStereoNet, self).__init__()
        # Create model
        with open(args.cfg, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

        #Decide Image Size
        if args.dataset == 'sceneflow':
            Img_size = yaml_cfg.get("IMG_SIZE").get("SCENEFLOW")
        elif args.dataset in ['kitti15','kitti12']: 
            Img_size = yaml_cfg.get("IMG_SIZE").get("KITTI")
        elif args.dataset == 'middlebury': 
            Img_size = yaml_cfg.get("IMG_SIZE").get("MIDDLEBURY")
        elif args.dataset in ['scared2019','scared2019_small']:
            Img_size = yaml_cfg.get("IMG_SIZE").get("SCARED")
        elif args.dataset == 'davinci':
            Img_size = yaml_cfg.get("IMG_SIZE").get("DVPN")

        self.maxdisp   = args.maxdisp
        self.height    = Img_size[0]
        self.width     = Img_size[1]
        self.zero_head = zero_head


        #parameters for feature net
        self.embed_dim_fea = yaml_cfg.get("MODEL").get("FEATURE_NET").get("EMBED_DIM")
        self.win_size_fea  = yaml_cfg.get("MODEL").get("FEATURE_NET").get("WINDOW_SIZE")
        self.enc_depth_fea = yaml_cfg.get("MODEL").get("FEATURE_NET").get("DEPTHS")
        self.num_heads_fea = yaml_cfg.get("MODEL").get("FEATURE_NET").get("NUM_HEADS")

        #parameters for matching net
        network_path_mat = np.load('pretrained_ckpt/leastereo_sf/architecture/matching_network_path.npy')
        cell_arch_mat    = np.load('pretrained_ckpt/leastereo_sf/architecture/matching_genotype.npy')
        print('Matching network path:{} \n'.format( network_path_mat))
        network_arch_mat = network_layer_to_space(network_path_mat)

        #define Feature parameters
        self.feature = FeatureNet(img_size=(self.height, self.width),
                                patch_size=(3, 3),
                                in_chans=3,
                                num_classes=32,
                                embed_dim=self.embed_dim_fea,
                                depths=self.enc_depth_fea,
                                num_heads=self.num_heads_fea,
                                window_size=self.win_size_fea,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

        #define Matching parameters
        self.matching = newMatching(network_arch_mat, cell_arch_mat) 
        self.disp = Disp(self.maxdisp)

    def forward(self, x, y): 
        x = self.feature(x)     
        y = self.feature(y) 

        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.maxdisp//3),  x.size()[2],  x.size()[3]).zero_() 
        for i in range(int(self.maxdisp//3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y

        cost = self.matching(cost) 
        disp0 = self.disp(cost)       
        return disp0

