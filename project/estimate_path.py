#!/usr/bin/env python
'''Return 2D parametric linear motion model
    at each instance of time.'''

import cv2
import numpy as np
from helper import drawMatches

def match_descriptors(prev_, curr_, threshold_=1.5):
    '''Given 2 descriptors returns the 2 indices corresponding
    to previous and curr, respectively using nearest neighbor calculated
    using L2 norm'''
    pass

def estimate_path(vid_, method='L2'):
    f_count, f_height, f_width, color_scale = vid_.shape
    sift = cv2.SIFT()
    prev_frame = vid_[0]
    prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)
    if method == 'L2': 
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        for i in range(1, f_count):
            next_frame = vid_[i]
            curr_kp, curr_des = sift.detectAndCompute(next_frame, None)
            matches = bf.match(prev_des, curr_des)
            # matches = bf.knnMatch(prev_des, curr_des, k=2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)

            # Draw first 10 matches.
            # img3 = drawMatches(vid_[0], prev_kp, vid_[1], curr_kp, matches[:10])

            break
    elif method == 'NN':
        # kNN used David Lowe's ratio test as described in
        # 7.1.) Keypoint matching of the SIFT paper

        # For some reason, this tends to be noiser when testing,
        # so the default is to use L2 distance norm and cutoff based on a threshold
        bf = cv2.BFMatcher()
        for i in range(1, f_count):
            next_frame = vid_[i]
            curr_kp, curr_des = sift.detectAndCompute(next_frame, None)
            matches = bf.knnMatch(prev_des, curr_des, k=2)
            # Sort them in the order of their distance.
            parsed_matches = []
            for m, n in matches:
                if m.distance < 0.80 * n.distance:
                    parsed_matches.append(m)

            # Draw first 10 matches.
            # img3 = drawMatches(vid_[0], prev_kp, vid_[1], curr_kp, parsed_matches[:10])

            break
