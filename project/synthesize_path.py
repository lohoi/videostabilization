#!/usr/bin/env python

import cv2
import numpy as np
from estimate_path import estimate_transform

def synthesize_path(vid_, p_opt, crop_ratio = 0.8):
    f_count, f_height, f_width, color_scale = vid_.shape

    x_center = f_width/2
    y_center = f_height/2

    x_cropped = f_width * crop_ratio
    y_cropped = f_height * crop_ratio

    # grayscale for now, color_scale == 1
    vid = vid_.reshape(f_count, f_height, f_width)
    vid_opt = vid;

    for i in range(1, f_count):
        # 3 x 3 stabiliation transform
        m_stable = p_opt[i-1][:, :]

        # m_stable = np.transpose(m_stable)
        x = []
        X = []

        # location of cropped + transformed corners (1 x 3)
        a = np.array([x_center - x_cropped/2, y_center - y_cropped/2, 1])
        b = np.array([x_center - x_cropped/2, y_center + y_cropped/2, 1])
        c = np.array([x_center + x_cropped/2, y_center + y_cropped/2, 1])
        d = np.array([x_center + x_cropped/2, y_center - y_cropped/2, 1])
        a = np.dot(np.transpose(a), m_stable)
        b = np.dot(np.transpose(b), m_stable)
        c = np.dot(np.transpose(c), m_stable)
        d = np.dot(np.transpose(d), m_stable)
        # input coordinates 4 x 2
        x = np.array([a[0:2], b[0:2], c[0:2], d[0:2]])

        # location of reconstructed corners
        A = np.array([0, 0])
        B = np.array([0, f_height])
        C = np.array([f_width, f_height])
        D = np.array([f_width, 0])
        # output coordinates 4 x 2
        X = np.array([A, B, C, D])
        height, width = X.shape
        # output coordinates 4 x 3
        X = np.append(X, np.ones((height,1)), axis=1)

        # estimate reconstuction transform  2 x 3
        m_recon = estimate_transform(x, X)
        vid_opt[i] = cv2.warpAffine(vid[i], m_recon, (f_width,f_height))


    vid_opt[0] = vid_opt[1];

    return vid_opt
