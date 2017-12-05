    #!/usr/bin/env python
import cv2
import numpy as np
from estimate_path import estimate_transform
from write_video import *

# Input:    original video
#           optimal path transform
#           crop ratio
def synthesize_path(vid_, p_opt, crop_ratio = 0.5):
    f_count, f_height, f_width, color_chan = vid_.shape
    x_center = int(f_width/2)
    y_center = int(f_height/2)

    x_cropped = int(f_width * crop_ratio)
    y_cropped = int(f_height * crop_ratio)

    # grayscale for now, color_scale == 1
    # vid = vid_.reshape(f_count, f_height, f_width)
    vid = vid_[:, :, :, 0].copy()
    vid_crop = np.zeros((f_count, y_cropped, x_cropped))
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

        # cropped corners static
        left_crop = int(x_center - x_cropped/2)
        right_crop = int(x_center + x_cropped/2)
        top_crop = int(y_center - y_cropped/2)
        bot_crop = int(y_center + y_cropped/2)

        # location of cropped corners 1 x 3
        # a = np.array([left_crop, top_crop, 1])
        # b = np.array([left_crop, bot_crop, 1])
        # c = np.array([right_crop, bot_crop, 1])
        # d = np.array([right_crop, top_crop, 1])

        a = np.array([top_crop, left_crop,  1])
        b = np.array([bot_crop, left_crop,  1])
        c = np.array([bot_crop, right_crop, 1])
        d = np.array([top_crop, right_crop, 1])

        # transform cropped corners 1 x 3
        a = np.dot(m_stable, np.transpose(a))
        b = np.dot(m_stable, np.transpose(b))
        c = np.dot(m_stable, np.transpose(c))
        d = np.dot(m_stable, np.transpose(d))

        # smooth transformed corners 4 x 2
        x = np.array([a[0:2], b[0:2], c[0:2], d[0:2]])

        # cropped corners transformed
        top = min(int(a[0]), 0)
        bot = max(int(c[0]), y_cropped)
        left = min(int(a[1]), 0)
        right = max(int(c[1]), x_cropped)

        print i
        print "{} {} x {} {}".format(top, bot, left, right)


        vid_crop[i][top:bot, left:right] = vid[i][top:bot, left:right]
        cv2.imshow('crop transform', vid_crop[i])
        cv2.waitKey(0)
            # cv2.warpAffine(vid[i][left:right, top:bot], m_stable_2x3, !!(x_cropped, y_cropped))


        # location of reconstructed corners
        A = np.array([0, 0])
        B = np.array([0, f_height])
        C = np.array([f_height, f_width])
        D = np.array([0, f_width])
        # output coordinates 4 x 2
        X = np.array([A, B, C, D])
        height, width = X.shape
        # output coordinates 4 x 3
        X = np.append(X, np.ones((height,1)), axis=1)

        # estimate reconstuction transform  2 x 3
        m_recon = estimate_transform(x, X)
        vid_recon[i] = cv2.warpAffine(vid[i], m_recon, (f_width,f_height))

        if i == 50:
            cv2.imwrite("../results/original_frame.png", vid_[i])
            cv2.imwrite("../results/smoothed_frame.png", vid_smooth[i])
            cv2.imwrite("../results/cropped_frame.png", vid_crop[i])
            cv2.imwrite("../results/reconstructed_frame.png", vid_recon[i])

    vid_recon[0] = vid_recon[1];
    write_video('../results/original.mp4', vid_)
    write_video('../results/smoothed.mp4', vid_smooth)
    write_video('../results/cropped.mp4', vid_crop)
    write_video('../results/reconstructed.mp4', vid_recon)

    return vid_recon
