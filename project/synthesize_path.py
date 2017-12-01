#!/usr/bin/env python

import cv2
import numpy as np

def synthesize_path(vid_, p_opt):
    f_count, f_height, f_width, color_scale = vid_.shape

    vid_opt = vid_;
    for i in range(1, f_count):
        m_affine = p_opt[0:1, :]
        warpAffine(vid_[i], vid_opt[i], p_opt[i-1], vid_opt[i].size())

    return vid_opt
