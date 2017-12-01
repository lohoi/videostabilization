#!/usr/bin/env python

import cv2
import numpy as np

def synthesize_path(vid_, p_opt):
    f_count, f_height, f_width, color_scale = vid_.shape
    vid = vid_.reshape(f_count, f_height, f_width)
    vid_opt = vid;
    for i in range(1, f_count):
        m_affine = p_opt[i-1][0:2, :]
        vid_opt[i] = cv2.warpAffine(vid[i], m_affine, (f_width,f_height))

        # warpAffine(vid_[i], vid_opt[i], m_affine, vid_opt[i].size())

    return vid_opt
