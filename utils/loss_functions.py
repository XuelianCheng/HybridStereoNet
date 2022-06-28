from __future__ import division
import torch
from torch import nn
from torch.autograd import Variable
import skimage.io
import numpy as np
import math
import pdb

def psnr_np(img1, img2):
    mse = np.mean((np.round(img1) - np.round(img2)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = np.ceil(img1.max()) #255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def createGrid(F):
    n, H, W,_ = F.size()  
    G = Variable(torch.FloatTensor(n,H,W,2).cuda())  
    G[:,:,:,0] = torch.linspace(-1,1,H).view(1,H,1).repeat(n,1,W)
    G[:,:,:,1] = torch.linspace(-1,1,W).view(1,1,W).repeat(n,H,1)
    G[:,:,:,0] = G[:,:,:,0].add(F[:,:,:,1]/(H-1)*2)
    G[:,:,:,1] = G[:,:,:,1].add(F[:,:,:,0]/(W-1)*2)
    return G
      
def generate_image_left(img, disp):
    ndisp = torch.stack([-disp, torch.zeros_like(disp)],-1)
    pixel_coords = createGrid(ndisp) 
    return torch.nn.functional.grid_sample(img.transpose(2,3), pixel_coords)    
  
def generate_image_right(img, disp):        
    ndisp = torch.stack([disp, torch.zeros_like(disp)],-1)
    pixel_coords = createGrid(ndisp) 
    return torch.nn.functional.grid_sample(img.transpose(2,3), pixel_coords)  
  
def photometric_reconstruction(left, right, disp_left_est,disp_right_est):  
    left_est  = generate_image_left(right, disp_left_est)  
    # right_est = generate_image_right(left, disp_right_est)
    return left_est
  