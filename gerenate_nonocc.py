from PIL import Image
import argparse
import numpy as np
import sys
import pdb
import cv2
import tifffile as tiff
sys.path.append('../') # add relative path
from utils.preprocess import  compute_left_occ_region

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
left  = np.array(Image.open(root+'img_left_1280/' +name+ '.png'))
right = np.array(Image.open(root+'img_right_1280/' +name+ '.png'))
disp  = tiff.imread(root+'/disp_left_1280/' +name+ '.tiff')
# donwsample attention by stride of 3
h, w, _ = left.shape

if __name__ == '__main__':
	# manually compute occluded region
	occ_mask = compute_left_occ_region(w, disp)
	# Note: code for computing the metrics can be found in module/loss.py
	valid_mask = np.logical_and(disp > 0.0, ~occ_mask)
    # save images
	# cv2.imwrite(root+'occ_left_1280/' +name+ '.png', occ_mask * 255)
	cv2.imwrite(root+'nonocc_left_1280/' +name+ '.png', valid_mask* 255)