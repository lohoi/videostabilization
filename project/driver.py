#!/usr/bin/env python
'''Run Video Stabilization Algorithm.'''
from read_video import *
from write_video import *
from estimate_path import *
from optimize_path import *
from synthesize_path import *
from helper import *

import numpy as np
import pickle


def main():
    '''Main Driver.'''

    # read video
    filename = '../media/test_vid_eric.mp4'
    vid = read_video(filename)

    # 1.) Estimate original camera path
    # F, C = estimate_path(vid, method='NN')

    # pickle.dump(F, open("F.p", "wb"))
    # pickle.dump(C, open("C.p", "wb"))

    F = pickle.load(open("F.p", "rb"))
    C = pickle.load(open("C.p", "rb"))

    # print F[0]

    # 2.) Estimate new camera path
    crop_ratio = 0.8
    B = optimizePathTransforms(F, vid.shape, crop_ratio)
    
    # path = []
    # F_cum = []
    # F_cum.append(F[0])
    # path.append(np.dot(F[0],B[0]))
    # for t in range(1,len(F)):
    #     F_cum.append(np.dot(F[t], F_cum[t-1]))
    #     path.append(np.dot(F_cum[t],B[t]))
    # plot_path(path)

    # 3.) Synthesize video with new camera path
    vid_opt = synthesize_path(vid, C)

    # write video
    write_video('../output.mp4', vid_opt)

if __name__ == "__main__":
    main()
