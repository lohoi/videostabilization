#!/usr/bin/env python
'''Return 2D parametric linear motion model
    at each instance of time.'''

import cv2
import numpy as np
from helper import drawMatches

def estimate_transform(matches_, X, Y):
    '''Input: list of match objects
    Return: a 2x3 affine transform matrix
    Basically uses inv(X'X)X'Y.
    to compute the affine transformation matrix'''
    return np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    

def estimate_path(vid_, method='NN'):
    # TODO: we can do an analysis on L2 vs NN here for SIFT
    f_count, f_height, f_width, color_scale = vid_.shape
    sift = cv2.SIFT()
    prev_frame = vid_[0]
    prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)
    bf = None
    if method == 'L2': 
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif method == 'NN':
        # kNN used David Lowe's ratio test as described in
        # 7.1.) Keypoint matching of the SIFT paper
        bf = cv2.BFMatcher()

        for i in range(1, f_count):
            next_frame = vid_[i]
            curr_kp, curr_des = sift.detectAndCompute(next_frame, None)

            matches = None
            parsed_matches = []
            if method == 'L2':
                matches = bf.match(prev_des, curr_des)
                # Sort them in the order of their distance.
                matches = sorted(matches, key=lambda x:x.distance)
                total = sum(match.distance for match in matches)
                for m in matches:
                    # a variant on the NN ratio test proposed by
                    # David Lowe to weed out noisy matches
                    if m.distance < 0.8 * total:
                        parsed_matches.append(m)
            elif method == 'NN':
                matches = bf.knnMatch(prev_des, curr_des, k=2)
                for m, n in matches:
                    if m.distance < 0.80 * n.distance:
                        # David Lowe's NN ratio test
                        parsed_matches.append(m)
    
            # drawMatches(vid_[0], prev_kp, vid_[1], curr_kp, parsed_matches)
            X = []
            Y = []
            for m in parsed_matches:
                X.append(prev_kp[m.queryIdx].pt)
                Y.append(curr_kp[m.trainIdx].pt)


            Y = np.array(Y)
            height1, width1 = Y.shape
            Y = np.append(Y, np.ones((height1,1)), axis=1)

            X = np.array(X)
            height2, width2 = X.shape
            # X = np.append(X, np.ones((height2,1)), axis=1)

            assert height1 == height2, 'estimate path: height mismatch'
            assert width1 == width2, 'estimate path: width mismatch'
            assert width1 == 2, 'estimate path: incorrect width'
            
            A = estimate_transform(parsed_matches, X, Y)
            A = np.append(A, np.ones((1,3)), axis=0)
            A[2,0] = 0
            A[2,1] = 0

            # print np.append(X[1],np.ones(1))
            # print A
            # print np.dot(A, np.append(X[0],np.ones(1), axis=0 ) )
            # print Y[0]
            # print A.shape
            
            prev_kp = curr_kp
            prev_des = curr_des
            break