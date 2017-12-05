#!/usr/bin/env python

import matlab.engine
import numpy as np
import pickle

def optimizePathTransforms(F, vid_shape, crop_ratio):
    solution = cvxopt.solvers.lp(minimization,G,h)#,A,b)
    pickle.dump(solution, open("optimization.p", "wb"))
    #solution = pickle.load(open("optimization.p", "rb"))
    print solution
    return []
