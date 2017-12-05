    #!/usr/bin/env python
import cv2
import numpy as np
from estimate_path import estimate_transform
from write_video import *

# Input:    original video
#           optimal path transform
#           crop ratio
def synthesize_path(vid_, p_opt, crop_ratio = 0.8):
    f_count, f_height, f_width, color_scale = vid_.shape
    print f_height
    print f_width
    x_center = f_width/2
    y_center = f_height/2

    x_cropped = f_width * crop_ratio
    y_cropped = f_height * crop_ratio

    # grayscale for now, color_scale == 1
    vid = vid_.reshape(f_count, f_height, f_width)
    vid_crop = np.zeros((f_count, f_height, f_width))
    vid_smooth = np.zeros((f_count, f_height, f_width))
    vid_recon = np.zeros((f_count, f_height, f_width))

    for i in range(1, f_count):
        # 3 x 3 stabiliation transform
        m_stable = p_opt[i-1][:, :]
        m_stable_2x3 = p_opt[i-1][0:2, :]
        vid_smooth[i] = cv2.warpAffine(vid[i], m_stable_2x3, (f_width,f_height))

        # m_stable = np.transpose(m_stable)
        x = []
        X = []

        # cropped corners
        left_crop = x_center - x_cropped/2
        right_crop = x_center + x_cropped/2
        top_crop = y_center - y_cropped/2
        bot_crop = y_center + y_cropped/2

        # vid_crop[i] = vid[i][]

        # location of cropped + transformed corners (1 x 3)
        a = np.array([left_crop, top_crop, 1])
        b = np.array([left_crop, bot_crop, 1])
        c = np.array([right_crop, top_crop, 1])
        d = np.array([right_crop, bot_crop, 1])
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
        vid_recon[i] = cv2.warpAffine(vid[i], m_recon, (f_width,f_height))

        if i == 1:
            cv2.imwrite("../original_frame.png", vid_[i])
            cv2.imwrite("../smoothed_frame.png", vid_smooth[i])
            cv2.imwrite("../reconstructed_frame.png", vid_recon[i])

    vid_recon[0] = vid_recon[1];
    write_video('../original.mp4', vid_)
    write_video('../smoothed.mp4', vid_smooth)

    return vid_recon
