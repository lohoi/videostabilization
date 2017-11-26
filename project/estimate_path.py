#!/usr/bin/env python
'''Return 2D parametric linear motion model
    at each instance of time.'''

import cv2
import numpy as np

def estimate_path(vid):
    f_count, f_height, f_width = vid.shape
    sift = cv2.xfeatures2d.SIFT_create()
    print 'vid shape:', vid.shape
    for frame in vid:
        kp, des = sift.detectAndCompute(frame, None)
        print kp.shape
        break
