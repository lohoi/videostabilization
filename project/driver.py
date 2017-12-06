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

    print "Reading video"
    # read video
    # filename = '../media/test_vid_eric.mp4'
    filename = '../media/test1.mp4'
    vid = read_video(filename)

    print "Creating camera path"
    F = estimate_path(vid, method='NN')

    # pickle.dump(F, open("F.p", "wb"))
    # pickle.dump(C, open("C.p", "wb"))

    # F = pickle.load(open("F.p", "rb"))
    # C = pickle.load(open("C.p", "rb"))
    # plot_path(F)
    # B = pickle.load(open("B_albert.p", "rb"))

    print "Estimating new camera path"
    crop_ratio = 0.8
    B = optimizePathTransforms(F, vid.shape, crop_ratio)
    print type(B)
    plot_new_path(F,B)

    print "Reconstructing original video with path"
    vid_opt = synthesize_path(vid, B)

    print "Writing video"
    #write_video('../output.mp4', vid_opt)

if __name__ == "__main__":
    main()
