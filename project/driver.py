#!/usr/bin/env python

from estimate_path import *
from smooth_path import *
from synthesize_path import *

def main():
	# read video
	video = read_video('/media/test_vid_eric.mp4')
	# 1.) Estimate original camera path
	estimate_path(video)
	# 2.) Estimate new camera path
	# 3.) Synthesize video with new camera path

if __name__ == "__main__":
    main()

