#!/usr/bin/env python

'''
Computes 2D parametric linear motion model (original camera path)
'''
import cv2
import numpy as np
from helper import *

# Our own linear least squares implementation to estimate affine transformation
def estimate_transform(X, Y):
    '''Input: list of match objects
    Return: a 2x3 affine transform matrix
    Basically uses inv(1X'X)X'Y.
    to compute the affine transformation matrix'''
    # Note: Using linear regression may be noisy b/c it
    # factors in the tails. Try RANSAC in future?
    return np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)

def estimate_path(vid_, method='NN'):
    f_count, f_height, f_width, color_scale = vid_.shape
    sift = cv2.SIFT()
    prev_frame = vid_[0]
    prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)
    bf = None
    F = []

    print 'num frames:', f_count

    if method == 'L2':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif method == 'NN':
        # kNN used David Lowe's ratio test as described in
        # 7.1.) Keypoint matching of the SIFT paper
        bf = cv2.BFMatcher()

    # Fine homography between consecutive frames
    for i in range(1, f_count):
        next_frame = vid_[i]
        curr_kp, curr_des = sift.detectAndCompute(next_frame, None)

        matches = None
        parsed_matches = []

        # We found that NN works better
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
                if m.distance < 0.1 * n.distance:
                    # David Lowe's NN ratio test
                    parsed_matches.append(m)
            if len(parsed_matches) < 10:
                for m, n in matches:
                    if m.distance < 0.2 * n.distance:
                        parsed_matches.append(m)
            if len(parsed_matches) < 10:
                for m, n in matches:
                    if m.distance < 0.3 * n.distance:
                        parsed_matches.append(m)
            if len(parsed_matches) < 10:
                for m, n in matches:
                    if m.distance < 0.5 * n.distance:
                        parsed_matches.append(m)
            if len(parsed_matches) < 10:
                print "Cannot find enough feature matches between frames"

        # if i == 150:
        #     draw_matches(vid_[0], prev_kp, vid_[1], curr_kp, parsed_matches)
        # print "Estimating path of frame {}".format(i)

        # We tried linear least squares, but results were not as good as RANSAC
        # X = []
        # Y = []
        # for m in parsed_matches:
        #     X.append(prev_kp[m.queryIdx].pt)
        #     Y.append(curr_kp[m.trainIdx].pt)
        #
        # Y = np.array(Y)
        # height1, width1 = Y.shape
        # Y = np.append(Y, np.ones((height1,1)), axis=1)
        #
        # X = np.array(X)
        # height2, width2 = X.shape
        # # X = np.append(X, np.ones((height2,1)), axis=1)
        #
        # assert height1 == height2, 'estimate path: height mismatch'
        # assert width1 == width2, 'estimate path: width mismatch'
        # assert width1 == 2, 'estimate path: incorrect width'
        #
        # A = estimate_transform(X, Y)
        # A = np.append(A, np.ones((1,3)), axis=0)
        # A[2,0] = 0
        # A[2,1] = 0
        # F.append(A)

        # Find transfrom from current to previous, which is frame-by-frame camera path (instead of image path)
        src_pts = np.float32([curr_kp[m.trainIdx].pt for m in parsed_matches]).reshape(-1,1,2)
        dst_pts = np.float32([prev_kp[m.queryIdx].pt for m in parsed_matches]).reshape(-1,1,2)
        M = cv2.estimateRigidTransform(src_pts, dst_pts, True)
        M = np.append(M, np.ones((1,3)), axis=0)
        M[0,0] = 0
        M[0,1] = 0
        F.append(M)

        # print np.append(X[1],np.ones(1))
        # print A
        # print np.dot(A, np.append(X[0],np.ones(1), axis=0 ) )
        # print Y[0]
        # print A.shape

        prev_kp = curr_kp
        prev_des = curr_des
    assert len(F) == f_count-1, 'estimate_path: frames mismatch'

    # C = [np.eye(3)]
    # for i, f in enumerate(F):
    #     C.append(np.dot(C[i], f))

    return F
