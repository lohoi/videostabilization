#!/usr/bin/env python
'''Return 2D parametric linear motion model
    at each instance of time.'''

def estimate_path(vid):
    f_count, f_height, f_width = vid.shape
    for frame in vid:
        pass