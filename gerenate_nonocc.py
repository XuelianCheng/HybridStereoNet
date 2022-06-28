from PIL import Image
import argparse
import numpy as np
import sys
import pdb
import cv2
import torch
import tifffile as tiff
sys.path.append('../') # add relative path
from utils.loss_functions import  compute_occ_region

def get_args_parser():
    parser = argparse.ArgumentParser('Set Name', add_help=False)
    parser.add_argument('--name', type=str, default='', help="filename")
    args = parser.parse_args()
    return args

opt = get_args_parser()
print(opt)

#Read image
root = './dataset/SCARED2019_small/'
name = opt.name
left  = np.array(Image.open(root+'img_left/' +name+ '.png'))
right = np.array(Image.open(root+'img_right/' +name+ '.png'))
disp_l= tiff.imread(root+'/disp_left/' +name+ '.tiff')
disp_r= tiff.imread(root+'/disp_right/'+name+ '.tiff')

if __name__ == '__main__':
	#compute occluded region by left and right check
	x,y=torch.from_numpy(disp_l).float(),torch.from_numpy(disp_r).float()

	occ_mask_l,_ = compute_occ_region(x.unsqueeze(0),y.unsqueeze(0))
	occ_mask = occ_mask_l.squeeze().numpy()

	valid_mask = np.logical_and(disp_l > 0.0, occ_mask<1)
    # save images
	cv2.imwrite(root+'occ_left/' +name+ '.png', occ_mask * 255)
	cv2.imwrite(root+'nonocc_left/' +name+ '.png', valid_mask* 255)