#!/usr/bin/env python

import cvxopt
import pickle

def optimizePathTransforms(F, vid_shape, crop_ratio):
    DoF = 6
    affine_weighting = [1, 1, 100, 100, 100, 100]
    w = [10, 1, 100]
    weighting
    # Set up the minimization equation
    minimization = np.zeros(4 * vid_shape[0] * DoF)
    for i in range(vid_shape[0]):
        for j in range(DoF):
            minimization[DoF * i + j] = w[0] * affine_weighting[j] 
            minimization[vid_shape[0] * DoF + DoF * i + j] = w[1] * affine_weighting[j] 
            minimization[2 * vid_shape[0] * DoF + DoF * i + j] = w[2] * affine_weighting[j] 
    minimization = cvxopt.matrix(minimization)
    G = []
    h = []
    #Smoothness Constraints
    for i in range(vid_shape[0] - 3): # Going from Frame to Frame
        # 
    for i in range(vid_shape[0] - 3,vid_shape[0]): # Getting the last few constraints with D P
        #
    for i in range(3 * DoF * vid_shape[0]):
        # Slack Variables are Positive Constraints
        slack = np.zeros(4 * DoF * vid_shape[0])
        slack[i] = -1
        G.append(slack.tolist())
        h.append(0)
    # Proximity Constraints
    
    # Inclusion constraints

    # Transpose G and turn it into a matrix for cvxopt to be in the correct form
    G = cvxopt.matrix(np.array(G).T)
    # Turn h into a matrix to be used in the correct form
    h = cvxopt.matrix(h)
    solution = cvxopt.solvers(minimization,G,h)
    pickle.dump(F, open("optimization.p", "wb"))
    #solution = pickle.load(open("optimization.p", "rb"))
    return []
