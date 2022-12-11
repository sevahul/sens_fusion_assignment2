#!/usr/bin/env python3

import cv2
import numpy as np
import sys

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import os

def NCC(img1, img2): 
    # ignore occlusions
    valid1 = np.where(img1 > 0.01)
    valid2 = np.where(img2 > 0.01)

    img1_float = img_as_float(img1)[valid1 and valid2]
    img2_float = img_as_float(img2)[valid1 and valid2]
    
    img1_float = (img1_float - img1_float.mean())/np.linalg.norm(img1_float)
    img2_float = (img2_float - img2_float.mean())/np.linalg.norm(img2_float)
    cor = np.corrcoef(img1_float, img2_float)
    return cor[1,0]

def MSE(img1, img2, normed = True, scale_factor=2000, inverted=True):
    valid1 = np.where(img1 > 0.01)
    valid2 = np.where(img2 > 0.01)

    img1_float = img_as_float(img1)[valid1 and valid2]
    img2_float = img_as_float(img2)[valid1 and valid2]

    # img1_float = img_as_float(img1)
    # img2_float = img_as_float(img2)
    if normed:
        img1_float = (img1_float - img1_float.mean())/np.linalg.norm(img1_float)
        img2_float = (img2_float - img2_float.mean())/np.linalg.norm(img2_float)
    mse_score = mean_squared_error(img1_float*scale_factor, img2_float*scale_factor)
    if inverted:
        mse_score = 1/mse_score
    return mse_score
 
def SSIM(img1, img2, normed = True):
    img1_float = img_as_float(img1)
    img2_float = img_as_float(img2)
    if normed:
        img1_float = (img1_float - img1_float.mean())/np.linalg.norm(img1_float)
        img2_float = (img2_float - img2_float.mean())/np.linalg.norm(img2_float)
    ssim_score = ssim(img1_float, img2_float, data_range=img1_float.max() - img1_float.min())
    return ssim_score

if __name__ == "__main__":
    
    # read gt disparity image
    output_folder = os.path.join("output", "naive", "Art")
    data_folder = os.path.join("data", "Art")
    img_gt_path = os.path.join(data_folder, "disp1.png")
    img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)

    # list of previously calculated disparity images for each window size
    my_images = ["output_w3_naive.png",
                 "output_w5_naive.png", 
                 "output_w7_naive.png", 
                 "output_w9_naive.png"]
    nccs = []
    mses = []
    ssims = []
    
    # calculate metrics
    for my_img in my_images:
        img_path = os.path.join(output_folder, my_img)
        img_my = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        nccs.append(NCC(img_my, img_gt))
        mses.append(MSE(img_my, img_gt))
        ssims.append(SSIM(img_my, img_gt))
    print("w_size:\t   3    5    7    9")
    print("NCC:\t", np.round(nccs, 2))
    print("MSE:\t", np.round(mses, 2))
    print("SSIM:\t", np.round(ssims, 2))
 
