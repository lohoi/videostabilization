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
    filename = '../media/test_selfie.mp4'
    vid = read_video(filename)
    [frames, height, width, chan] = vid.shape

    # Step 1
    print "Creating camera path"
    F, C = estimate_path(vid, method='NN')

    # Step 2
    print "Estimating new camera path"
    crop_ratio = 0.8
    B = optimizePathTransforms(F, vid.shape, crop_ratio)

    print "Plotting camera paths - close window to continue"
    plot_new_path(F, B)

    # Step 3
    print "Reconstructing original video with path"
    vid_opt = synthesize_path(vid, B, 0.8)

    print "Writing video.."

    print "Complete!"


if __name__ == "__main__":
    main()
