#!/usr/bin/env python

'''Reconstruct video'''

import cv2
import numpy as np
from estimate_path import estimate_transform
from write_video import *

# Input:    original video
#           optimal path transform
#           crop ratio
def synthesize_path(vid_, p_opt, crop_ratio = 0.8):
    f_count, f_height, f_width, color_chan = vid_.shape
    x_center = int(f_width/2)
    y_center = int(f_height/2)

    cropped_width = int(f_width * crop_ratio)
    cropped_height = int(f_height * crop_ratio)

    # grayscale
    vid = vid_[:, :, :, 0].copy()
    vid_crop = np.zeros((f_count, f_height, f_width))
    vid_recon = np.zeros((f_count, f_height, f_width))

    for i in range(1, f_count):

        # 3 x 3 stabiliation transform
        m_stable = p_opt[i-1][:, :]
        m_stable_2x3 = p_opt[i-1][0:2, :]

        crop_corns = []
        orig_corns = []

        # crop from center
        left_crop = int(x_center - cropped_width/2)
        right_crop = int(x_center + cropped_width/2)
        top_crop = int(y_center - cropped_height/2)
        bot_crop = int(y_center + cropped_height/2)

        a = np.array([top_crop, left_crop,  1])
        b = np.array([bot_crop, left_crop,  1])
        c = np.array([bot_crop, right_crop, 1])
        d = np.array([top_crop, right_crop, 1])

        # transform cropped corners 1 x 3
        a = np.dot(m_stable, np.transpose(a))
        b = np.dot(m_stable, np.transpose(b))
        c = np.dot(m_stable, np.transpose(c))
        d = np.dot(m_stable, np.transpose(d))

        crop_corns = [a[0:2].tolist(), b[0:2].tolist(), c[0:2].tolist(), d[0:2].tolist()]

        # cropped corners transformed
        top = max(int(a[0]), 0)
        bot = min(int(c[0]), f_height)
        left = max(int(a[1]), 0)
        right = min(int(c[1]), f_width)

        vid_crop[i] = vid[i].copy();
        cv2.rectangle(vid_crop[i], (left, top), (right,bot), (255, 0, 0) ,3);

        # location of original corners of video
        A = [0.0, 0.0]
        B = [float(f_height), 0.0]
        C = [float(f_height), float(f_width)]
        D = [0.0, float(f_width)]

        # original video corner points 4 x 2
        orig_corns = [A, B, C, D]

        m_recon, mask = cv2.findHomography(np.array(crop_corns), np.array(orig_corns), cv2.RANSAC)
        vid_recon[i] = cv2.warpPerspective(vid[i], m_recon, (f_width,f_height))

        if i == 100 or i == 101:
            cv2.imwrite("../results/original_frame{}.png".format(i), vid_[i])
            cv2.imwrite("../results/cropped_frame{}.png".format(i), vid_crop[i])
            cv2.imwrite("../results/reconstructed_frame{}.png".format(i), vid_recon[i])

    vid_recon[0] = vid_recon[1];
    write_video('../results/original.mp4', vid_)
    write_video('../results/cropped.mp4', vid_crop)
    write_video('../results/reconstructed.mp4', vid_recon)

    return vid_recon
