#!/usr/bin/env python

#import matlab.engine
import csv
import numpy as np
import pickle

def optimizePathTransforms(F, vid_shape, crop_ratio):
    with open('dim.csv', 'wb') as csvfile:
        dim_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        dim_writer.writerow([vid_shape[0]] + [vid_shape[1]] + [vid_shape[2]])
    with open('F.csv', 'wb') as csvfile:
        f_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for f in F:
            f_writer.writerow([f[0,0]] + [f[0,1]] +  [f[0,2]] + [f[0,0]] + [f[1,1]] +  [f[1,2]] + [f[2,0]] + [f[2,1]] +  [f[2,2]]) 
    #pickle.dump(solution, open("optimization.p", "wb"))
    #solution = pickle.load(open("optimization.p", "rb"))
    print solution
    return []
