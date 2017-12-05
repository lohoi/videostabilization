#!/usr/bin/env python
'''Return a N x 2 x 3 matrix of translation transformations'''
import cv2
import numpy as np
from helper import *

def create_pan_path(vid_):
    f_count = vid_.shape[0]
    F = []
    C = []

    r_pan = 0.0;
    c_pan = 0.0; # keep constant
    # each frame will be translated in the x direction
    for i in range(0, f_count):
        m = np.array([[1.0, 0.0, r_pan], [0.0, 1.0, c_pan], [0.0, 0.0, 1.0]])
        c_pan = c_pan + 1.0
        F.append(m)

    # F is the transformation matrix for each frame
    # C is estimated path (unused)
    return F, C
