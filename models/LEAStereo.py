import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb

from models.leastereo.decoding_formulas import network_layer_to_space
from models.leastereo.new_model_2d import newFeature
from models.leastereo.skip_model_3d import newMatching

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

class LEAStereo(nn.Module):
    def __init__(self, args):
        super(LEAStereo, self).__init__()

        network_path_fea = np.load('pretrained_ckpt/leastereo_sf/architecture/feature_network_path.npy')
        cell_arch_fea    = np.load('pretrained_ckpt/leastereo_sf/architecture/feature_genotype.npy')
        network_path_mat = np.load('pretrained_ckpt/leastereo_sf/architecture/matching_network_path.npy')
        cell_arch_mat    = np.load('pretrained_ckpt/leastereo_sf/architecture/matching_genotype.npy')
        print('Feature network path:{}\nMatching network path:{} \n'.format(network_path_fea, network_path_mat))

        network_arch_fea = network_layer_to_space(network_path_fea)
        network_arch_mat = network_layer_to_space(network_path_mat)

        self.maxdisp = args.maxdisp
        self.feature = newFeature(network_arch_fea, cell_arch_fea)
        self.matching= newMatching(network_arch_mat, cell_arch_mat) 
        self.disp = Disp(self.maxdisp)

    def forward(self, x, y):
        x = self.feature(x)       
        y = self.feature(y) 

        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.maxdisp/3),  x.size()[2],  x.size()[3]).zero_() 
        for i in range(int(self.maxdisp/3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y
        
        cost = self.matching(cost)     
        disp = self.disp(cost)   
        return disp

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
                         + list(self.decoder.parameters()) \
                         + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params
