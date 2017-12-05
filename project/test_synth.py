#!/usr/bin/env python
import numpy as np
import skvideo.io

from read_video import *
from write_video import *
from estimate_path import *
from smooth_path import *
from synthesize_path import *
from pan_test import *

import pickle


# filename = '../media/test_vid_eric.mp4'
filename = '../media/still_vid.mp4'
vid = read_video(filename)

# F, C = estimate_path(vid, method='NN')
# F = pickle.load(open("F.p", "rb"))
# C = pickle.load(open("C.p", "rb"))
F, C = create_pan_path(vid)

vid_recon = synthesize_path(vid, F)
