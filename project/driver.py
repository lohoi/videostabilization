#!/usr/bin/env python
'''Run Video Stabilization Algorithm.'''
from read_video import *
from estimate_path import *
from smooth_path import *
from synthesize_path import *


def main():
    '''Main Driver.'''

    # read video
    filename = '../media/test_vid_eric.mp4'
    vid = read_video(filename)

    # 1.) Estimate original camera path
    F = estimate_path(vid, method='NN')
    
    # 2.) Estimate new camera path
    # 3.) Synthesize video with new camera path

if __name__ == "__main__":
    main()
