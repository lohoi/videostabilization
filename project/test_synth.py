#!/usr/bin/env python
import numpy as np
import skvideo.io

from read_video import *
from write_video import *
from estimate_path import *
from smooth_path import *
from synthesize_path import *

import pickle


filename = '../media/test_vid_eric.mp4'
vid = read_video(filename)
write_video('../test.mp4', vid)

# F, C = estimate_path(vid, method='NN')
F = pickle.load(open("F.p", "rb"))
C = pickle.load(open("C.p", "rb"))
vid_recon = synthesize_path(vid, F)
write_video('../result.mp4', vid_recon)
