import argparse
import io
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path

import cv2
import cv2 as cv
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import tqdm
import zarr
from PIL import Image, ImageDraw
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           grey_closing, grey_dilation, grey_erosion)
from skimage import feature, filters
from skimage.color import rgb2gray
from skimage.data import camera
from skimage.util import compare_images
from omegaconf import OmegaConf


cfg = OmegaConf.load('conf/config.yml')

def plot_images(*imgs, axis=0, raw=False):
    if axis == 0 :
        nrows, ncols = len(imgs), 1
    else:
        nrows, ncols = 1, len(imgs)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                            figsize=(12, 8*len(imgs)))

    axes = axes.ravel()
    for i, img in enumerate(imgs):
        axes[i].imshow(img, cmap='gray' if i != 0 else None)

def detect_image(file):
    arr = imageio.imread(file)
    gray = rgb2gray(arr)
    # hes =  filters.hessian(gray)
    edge = filters.prewitt(gray)
    binary = np.where(edge > np.quantile(edge, 0.9), edge.max(), edge.min())
    closed = binary_closing(binary, iterations=15)
    return closed



src_dir = Path(cfg.catalogue.img_dir)
out_dir = Path(cfg.catalogue.out_dir)

files = list(
    src_dir.glob('**/*.jpg')
    )

dts = pd.to_datetime(
    [file.stem.split('_')[-1] for file in files],
    format='%Y%m%d%H%M%S'
)



file_dts = [
    (file, dt) for file, dt 
    in zip(files, dts) 
    if (dt.hour >= 7 and dt.hour <= 18)
    ]

rows = []

for file, dt in tqdm.tqdm(file_dts):
    pass
    img = cv.imread(file.as_posix())
    grey = cv.imread(file.as_posix(),0)    
    
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(grey)

    # cl1 = cv2.GaussianBlur(cl1,(3,3), SigmaX=0, SigmaY=0) 
    dst = cv2.Canny(image=cl1, threshold1=100, threshold2=200)
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)    
    for i in range(0, len(linesP)):
        l = linesP[i][0]

        p1_x, p1_y, p2_x, p2_y = linesP[i][0]
        p1 = (p1_x, p1_y)
        p2 = (p2_x, p2_y)
        cv.line(cdstP, p1, p2, (255,0,0), 3, cv.LINE_AA)
        # cv.line(img, p1, p2, (255,0,0), 3, cv.LINE_AA)

    hl_series = pd.Series(
            np.mean(cdstP[...,0] == 255, axis=1)
        )
    # plt.imshow(cdstP)    
    hl_unnormalized = hl_series.idxmax()    
    rows.append((file, dt, hl_unnormalized))

table = pd.DataFrame(rows, columns = ['file', 'datetime', 'hl'])

upper_lim = table.hl.quantile(0.9) + 2.5 * table.hl.std()
lower_lim = table.hl.quantile(0.1) - 2.5 * table.hl.std()

# qc extreme values
table.loc[
    table.hl.gt(upper_lim) | table.hl.le(lower_lim),
    'hl'] = np.nan


rolling_median = (
    table
    .set_index('datetime')
    .sort_index()
    .rolling(cfg.params.window)
    ['hl']
    .median()
    .round()
    .astype('Int64')
    .to_frame(name='rolling_median')
)

table = (
    table
    .merge(rolling_median, left_on=['datetime'], right_index=True)
    .dropna()
    .astype( {'hl': int})
)

for tup in table.itertuples(index=False):
    img = cv.imread(tup.file.as_posix())
    height, width, _ = img.shape
    image = cv2.line(img, (0, tup.hl), (width-1, tup.hl), (0,0,255), 5)
    img = cv2.line(img, (0, tup.rolling_median), (width-1, tup.rolling_median), (255,0,0), 5)
    imageio.imsave(out_dir / tup.file.name, img)





    
    




