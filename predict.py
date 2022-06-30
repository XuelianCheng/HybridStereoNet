from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10

import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.LEAStereo import LEAStereo
from models.build_model_hybrid import HybridStereoNet

from config_utils.predict_args import obtain_predict_args
from utils.colorize import get_color_map
from utils.multadds_count import count_parameters_in_MB, comp_multadds
from utils.util_functions import photometric_reconstruction, psnr_np
from skimage.metrics import structural_similarity as ssim
from time import time
from struct import unpack
import matplotlib.pyplot as plt
import re
import numpy as np
import pdb
import cv2
from path import Path

opt = obtain_predict_args()
print(opt)

torch.backends.cudnn.benchmark = True

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Building LEAStereo model')
model = LEAStereo(opt)

# print('===> Building HybridStereoNet model')
# model = HybridStereoNet(opt) 

print('Total Params = %.2fMB' % count_parameters_in_MB(model))
print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))
   
mult_adds = comp_multadds(model, input_size=(3,opt.crop_height, opt.crop_width)) #(3,192, 192))
print("compute_average_flops_cost = %.2fMB" % mult_adds)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        state_dict = checkpoint['state_dict']   
        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]
        model.load_state_dict(checkpoint['state_dict'], strict=False)      
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

turbo_colormap_data = get_color_map()

def RGBToPyCmap(rgbdata):
    nsteps = rgbdata.shape[0]
    stepaxis = np.linspace(0, 1, nsteps)

    rdata=[]; gdata=[]; bdata=[]
    for istep in range(nsteps):
        r = rgbdata[istep,0]
        g = rgbdata[istep,1]
        b = rgbdata[istep,2]
        rdata.append((stepaxis[istep], r, r))
        gdata.append((stepaxis[istep], g, g))
        bdata.append((stepaxis[istep], b, b))

    mpl_data = {'red':   rdata,
                 'green': gdata,
                 'blue':  bdata}

    return mpl_data

mpl_data = RGBToPyCmap(turbo_colormap_data)
plt.register_cmap(name='turbo', data=mpl_data, lut=turbo_colormap_data.shape[0])

def readPFM(file): 
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)

    return img, height, width

def save_pfm(filename, image, scale=1):
    '''
    Save a Numpy array to a PFM file.
    '''
    color = None
    file = open(filename, "w")
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)

def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width: 
        # padding zero 
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp    
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]	
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data

def test_scared(leftname, rightname, savename, imgname):

    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    torch.cuda.synchronize()
    start_time = time()
    with torch.no_grad():
        prediction = model(input1, input2)
    torch.cuda.synchronize()
    end_time = time()
    
    print("Processing time: {:.4f}".format(end_time - start_time))
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height or width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    # temp = np.flipud(temp)

    if opt.scared2019_small:
    # if opt.scared2019_small or opt.scared2019:
        imgpath = os.path.dirname(imgname)
        if not os.path.exists(imgpath):
            os.makedirs(imgpath)
        plot_disparity(imgname, temp, 263)

    disppath = os.path.dirname(savename)
    if not os.path.exists(disppath):
        os.makedirs(disppath)
    cv2.imwrite(savename,temp)

def test_davinci(leftname, rightname, savename, imgname):

    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    torch.cuda.synchronize()
    start_time = time()
    with torch.no_grad():
        prediction_l = model(input1, input2)
        prediction_r = model(input2, input1)
    torch.cuda.synchronize()
    end_time = time()

    input1_pred = photometric_reconstruction(input1, input2, prediction_l, prediction_r)

    left = input1.squeeze(0).cpu().detach().numpy()
    pred_left = input1_pred.squeeze(0).cpu().detach().numpy()
    pred_disp_l = prediction_l.cpu().detach().numpy()
    if height <= opt.crop_height or width <= opt.crop_width:
        left = left[:, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
        pred_left = pred_left[:, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
        pred_disp_l = pred_disp_l[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        left = left[:, :, :]
        pred_left = pred_left[:, :, :]
        pred_disp_l = pred_disp_l[0, :, :]
    

    ssim_r = ssim(left[0, :, :], pred_left[0, :, :])
    ssim_g = ssim(left[1, :, :], pred_left[1, :, :])
    ssim_b = ssim(left[2, :, :], pred_left[2, :, :])
    wssim = 1/3 * (ssim_r + ssim_g + ssim_b)

    psnr = psnr_np(left, pred_left)
    print("Processing time: {:.4f} ssim: {:.4f} psnr: {:.4f}".format(end_time - start_time, wssim, psnr))

    # imgpath = os.path.dirname(imgname)
    # if not os.path.exists(imgpath):
    #     os.makedirs(imgpath)
    # plot_disparity(imgname, temp, 192)

    # disppath = os.path.dirname(savename)
    # if not os.path.exists(disppath):
    #     os.makedirs(disppath)
    # cv2.imwrite(savename,temp)
    return wssim, psnr

def plot_disparity(savename, data, max_disp):
    plt.imsave(savename, data, vmin=0, vmax=max_disp, cmap='turbo')

   
if __name__ == "__main__":
    ssim_all = 0
    pnsr_all = 0
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    for index in range(len(filelist)):
        current_file = filelist[index]

        if opt.scared2019:
            leftname = file_path + current_file[0: len(current_file) - 1]
            img_path = Path(opt.save_path + 'images')
            img_path.makedirs_p()
            temppath = current_file[0: len(current_file) - 1].split("/")
            if 'Left_Image' in leftname:
                rightname = leftname.replace('Left_Image','Right_Image')
                savename = opt.save_path + temppath[0] +'/'+ temppath[1] +'/'+ temppath[2] +'/'+ temppath[-1].replace('Left_Image.png','left_disp_map.tiff')
                img_name = img_path +'/'+ temppath[0] +'/'+ temppath[1] +'/'+ temppath[2] +'/'+ temppath[-1].replace('Left_Image','left_disp_map')
            elif 'left_finalpass' in leftname:
                rightname = leftname.replace('left_finalpass','right_finalpass')
                savename = opt.save_path + temppath[0] +'/'+ temppath[1] +'/'+ temppath[2] +'/'+ temppath[-1].replace('.png','.tiff')
                img_name = img_path +'/'+ temppath[0] +'/'+ temppath[1] +'/'+ temppath[2] +'/'+ temppath[-1]
            test_scared(leftname, rightname, savename, img_name)

        if opt.scared2019_small:
            leftname = file_path + current_file[0: len(current_file) - 1]
            rightname = leftname.replace('img_left','img_right')
            img_path = Path(opt.save_path + 'images')
            img_path.makedirs_p()
            temppath = current_file[0: len(current_file) - 1].split("/")
            savename = opt.save_path + temppath[1].replace('.png','.tiff')
            img_name = img_path +'/'+ temppath[1]
            test_scared(leftname, rightname, savename, img_name)

        if opt.davinci:
            leftname = file_path + current_file[0: len(current_file) - 1]
            rightname = leftname.replace('image_0','image_1')
            img_path = Path(opt.save_path + 'images')
            img_path.makedirs_p()
            temppath = current_file[0: len(current_file) - 1].split("/")
            savename = opt.save_path + temppath[2].replace('.png','.tiff')
            img_name = img_path +'/'+ temppath[2]
            ssim_ep, psnr_ep = test_davinci(leftname, rightname, savename, img_name)

            ssim_all += ssim_ep
            pnsr_all += psnr_ep

    if opt.davinci:
        print("ssim_avg: {:.4f} pnsr_avg: {:.4f}".format(ssim_all/len(filelist), pnsr_all/len(filelist)))


