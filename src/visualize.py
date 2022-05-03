import numpy as np
from PIL import Image
import gradio as gr
import argparse
import os
import cv2 as cv
import matplotlib.pyplot as plt


def sepia(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(grey)

    dst = cv.Canny(image=cl1, threshold1=100, threshold2=200)
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)    
    for i in range(0, len(linesP)):
        l = linesP[i][0]

        p1_x, p1_y, p2_x, p2_y = linesP[i][0]
        p1 = (p1_x, p1_y)
        p2 = (p2_x, p2_y)
        cv.line(cdstP, p1, p2, (255,0,0), 3, cv.LINE_AA)

    
    hl = np.argmax(np.mean(cdstP[...,0] == 255, axis=1))
    height, width, _ = img.shape
    image = cv.line(img, (0, hl), (width-1, hl), (0,0,255), 5)
    return image

iface = gr.Interface(sepia, gr.inputs.Image(shape=(1280, 720)), "image")
iface.launch()






