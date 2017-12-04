#!/usr/bin/env python

import cvxopt
import pickle

def optimizePathTransforms(F, vid_shape, crop_ratio):
    DoF = 6
    # Set up the minimization equation
    minimization = np.zeros(4 * vid_shape[0] * DoF)
    
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

    G = cvxopt.matrix(np.array(G).T)
    h = cvxopt.matrix(h)
    solution = cvxopt.solvers(minimization,G,h)
    pickle.dump(F, open("optimization.p", "wb"))
    #solution = pickle.load(open("optimization.p", "rb"))
    return []
