#!/usr/bin/env python
'''write 3D np array into video'''
import skvideo.io
import numpy as np

def write_video(filename, buf=None):
    if buf is None:
        buf = np.random.random(size=(5, 480, 680, 3)) * 255
        buf = outputdata.astype(np.uint8)

    skvideo.io.vwrite(filename, buf)
