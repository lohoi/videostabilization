#!/usr/bin/env python
'''Return a N x 2 x 3 matrix of translation transformations'''
import cv2
import numpy as np
from helper import *

def create_pan_path(vid_):
    f_count, f_height, f_width, color_scale = vid_.shape
    F = []
    C = []

    x_pan = 0;
    y_pan = 0; # keep constant
    # each frame will be translated in the x direction
    for i in range(f_count):
        m = np.array([1, 0, x_pan], [0, 1, 0])
        x_pan = x_pan + 5
        F.append(m)

    # F is the transformation matrix for each frame
    # C is estimated path (unused)
    return F, C
