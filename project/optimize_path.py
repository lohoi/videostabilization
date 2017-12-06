#!/usr/bin/env python

#import matlab.engine
import csv
import numpy as np
import pickle
import subprocess

def optimizePathTransforms(F, vid_shape, crop_ratio):
    # Writes the Dim and F to a CSV to be called by the matlab code
    with open('dim.csv', 'wb') as csvfile:
        dim_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        dim_writer.writerow([vid_shape[0]] + [vid_shape[1]] + [vid_shape[2]])
    with open('F.csv', 'wb') as csvfile:
        f_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for f in F:
            f_writer.writerow([f[0,0]] + [f[0,1]] +  [f[0,2]] + [f[0,0]] + [f[1,1]] +  [f[1,2]] + [f[2,0]] + [f[2,1]] +  [f[2,2]])

    # Call matlab code
    print "Calling Matlab CVX"
    subprocess.call("/Applications/MATLAB.app/bin/matlab -r matlab_opt_transform rm -nodisplay", shell=True)

    # Read resulting csv from matlab code
    p = []
    with open('p_ishit.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            row_count = 0
            p_val = np.zeros((3,3))
            for ind, val in enumerate(row):
                p_val[int(ind / 3), ind % 3] = val
            p_val[2,2] = 1
            p.append(p_val)
    # pickle.dump(p, open("p.p", "wb"))
    # p = pickle.load(open("p.p", "rb"))
    return p
