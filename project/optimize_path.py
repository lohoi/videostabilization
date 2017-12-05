#!/usr/bin/env python

import cvxopt
import numpy as np
import pickle

def getRt(F, f_len, DoF, i, e):
    R_t = []
    h_t = []
    e_t = []

    # constraint B(0, 0)
    temp = np.zeros(4 * f_len * DoF)
    temp[3 * f_len + i * DoF + 2] = F[i + 1][0, 0] - 1
    temp[3 * f_len + i * DoF + 4] = F[i + 1][0, 1]
    R_t.append(temp)
    h_t.append(0)
    e_t.append([e * f_len + i * DoF + 2, -1])
    R_t.append((-1 * temp))
    h_t.append(0)
    e_t.append([e * f_len + i * DoF + 2, 1])

    # constraint B(0, 1)
    temp = np.zeros(4 * f_len * DoF)
    temp[3 * f_len + i * DoF + 3] = F[i + 1][0, 0] - 1
    temp[3 * f_len + i * DoF + 5] = F[i + 1][0, 1]
    R_t.append(temp)
    h_t.append(0)
    e_t.append([e * f_len + i * DoF + 3, -1])
    R_t.append((-1 * temp))
    h_t.append(0)
    e_t.append([e * f_len + i * DoF + 3, 1])

    # constraint B(0, 2)
    temp = np.zeros(4 * f_len * DoF)
    temp[3 * f_len + i * DoF] = F[i + 1][0, 0] - 1
    temp[3 * f_len + i * DoF + 1] = F[i + 1][0, 1]
    R_t.append(temp)
    h_t.append(-F[i + 1][0,2])
    e_t.append([e * f_len + i * DoF, -1])
    R_t.append((-1 * temp))
    h_t.append(F[i + 1][0,2])
    e_t.append([e * f_len + i * DoF, 1])

    # constraint B(1, 0)
    temp = np.zeros(4 * f_len * DoF)
    temp[3 * f_len + i * DoF + 2] = F[i + 1][1, 0] - 1
    temp[3 * f_len + i * DoF + 4] = F[i + 1][1, 1]
    R_t.append(temp)
    h_t.append(0)
    e_t.append([e * f_len + i * DoF + 4, -1])
    R_t.append((-1 * temp))
    h_t.append(0)
    e_t.append([e * f_len + i * DoF + 4, 1])

    # constraint B(1, 1)
    temp = np.zeros(4 * f_len * DoF)
    temp[3 * f_len + i * DoF + 3] = F[i + 1][1, 0] - 1
    temp[3 * f_len + i * DoF + 5] = F[i + 1][1, 1]
    R_t.append(temp)
    h_t.append(0)
    e_t.append([e * f_len + i * DoF + 5, -1])
    R_t.append((-1 * temp))
    h_t.append(0)
    e_t.append([e * f_len + i * DoF + 5, 1])

    # constraint B(1, 2)
    temp = np.zeros(4 * f_len * DoF)
    temp[3 * f_len + i * DoF] = F[i + 1][1, 0] - 1
    temp[3 * f_len + i * DoF + 1] = F[i + 1][1, 1]
    R_t.append(temp)
    h_t.append(-F[i + 1][1,2])
    e_t.append([e * f_len + i * DoF + 1, -1])
    R_t.append((-1 * temp))
    h_t.append(F[i + 1][1,2])
    e_t.append([e * f_len + i * DoF + 1, 1])

    return R_t, h_t, e_t

def optimizePathTransforms(F, vid_shape, crop_ratio):
    DoF = 6
    f_len = 4 #len(F)
    affine_weighting = [1, 1, 100, 100, 100, 100]
    w = [10, 1, 100]
    # Set up the minimization equation
    minimization = np.zeros(4 * f_len * DoF)
    for i in range(f_len):
        for j in range(DoF):
            minimization[DoF * i + j] = w[0] * affine_weighting[j] 
            minimization[f_len * DoF + DoF * i + j] = w[1] * affine_weighting[j] 
            minimization[2 * f_len * DoF + DoF * i + j] = w[2] * affine_weighting[j] 
    minimization = cvxopt.matrix(minimization)
    G = []
    h = []
    #Smoothness Constraints
    for i in range(f_len - 3): # Going from Frame to Frame
        R_t, h_t, e_t = getRt(F, f_len, DoF, i, 0)
        R_t1, h_t1, e_t1 = getRt(F, f_len, DoF, i + 1, 1)
        R_t2, h_t2, e_t2 = getRt(F, f_len, DoF, i + 2, 2)
        # e1 constraints
        for ind, constraint in enumerate(R_t):
            temp = constraint.tolist()
            temp[e_t[ind][0]] = e_t[ind][1]
            G.append(temp)
            h.append(h_t[ind])
        # e2 constraints
        for ind in range(len(R_t1)):
            temp = (R_t1[ind] - R_t[ind]).tolist()
            temp[e_t1[ind][0]] = e_t1[ind][1]
            G.append(temp)
            h.append(h_t1[ind] - h_t[ind])
        # e3 constraints
        for ind in range(len(R_t1)):
            temp = (R_t2[ind] - 2 * R_t1[ind] + R_t[ind]).tolist()
            temp[e_t2[ind][0]] = e_t2[ind][1]
            G.append(temp)
            h.append(h_t1[ind] - h_t[ind])
    #for i in range(f_len - 3,f_len): # Getting the last few constraints with D P
    # Slack Variables are Positive Constraints
    for i in range(3 * DoF * f_len):
        slack = np.zeros(4 * DoF * f_len)
        slack[i] = -1
        G.append(slack.tolist())
        h.append(0)
    # Proximity Constraints
    for i in range(f_len):
        # a_t, d_t constraints
        # a_t
        temp = np.zeros(4 * f_len * DoF)
        temp[3 * DoF * f_len + i * DoF + 2] = 1
        G.append(temp.tolist())
        h.append(1.1)
        G.append((-1 * temp).tolist())
        h.append(-0.9)
        # d_t
        temp = np.zeros(4 * f_len * DoF)
        temp[3 * DoF * f_len + i * DoF + 5] = 1
        G.append(temp.tolist())
        h.append(1.1)
        G.append((-1 * temp).tolist())
        h.append(0.9)
        # b_t, c_t constraints
        # b_t
        temp = np.zeros(4 * f_len * DoF)
        temp[3 * DoF * f_len + i * DoF + 3] = 1
        G.append(temp.tolist())
        h.append(0.1)
        G.append((-1 * temp).tolist())
        h.append(0.1)
        # c_t
        temp = np.zeros(4 * f_len * DoF)
        temp[3 * DoF * f_len + i * DoF + 4] = 1
        G.append(temp.tolist())
        h.append(0.1)
        G.append((-1 * temp).tolist())
        h.append(0.1)
        # b_t + c_t constraints
        temp = np.zeros(4 * f_len * DoF)
        temp[3 * DoF * f_len + i * DoF + 3] = 1
        temp[3 * DoF * f_len + i * DoF + 4] = 1
        G.append(temp.tolist())
        h.append(0.05)
        G.append((-1 * temp).tolist())
        h.append(0.05)
        #a_t - d_t constraints
        temp = np.zeros(4 * f_len * DoF)
        temp[3 * DoF * f_len + i * DoF + 2] = 1
        temp[3 * DoF * f_len + i * DoF + 5] = -1
        G.append(temp.tolist())
        h.append(0.1)
        G.append((-1 * temp).tolist())
        h.append(0.1)
    # Inclusion constraints

    # Transpose G and turn it into a matrix for cvxopt to be in the correct form
    print "G Len: ", len(G)
    print "G[0] Len: ", len(G[0])
    G = cvxopt.matrix(np.array(G))
    # Turn h into a matrix to be used in the correct form
    h = cvxopt.matrix(h)
    print "F Len: ", len(F)
    print "num_columns ", 6 * len(F) * 4
    print "Min Size: ", minimization.size
    print "G Size: ", G.size
    print "H Size: ", h.size
    solution = cvxopt.solvers.lp(minimization,G,h)
    pickle.dump(F, open("optimization.p", "wb"))
    #solution = pickle.load(open("optimization.p", "rb"))
    return []
