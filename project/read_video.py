#!/usr/bin/env python
'''Read video and returns a 3D np array'''
import cv2
import numpy as np
import skvideo.io

def read_video(filename, isGray=True):
    # Return np array. Default is to return a grayscale for easier computation
    # cap = cv2.VideoCapture(filename)
    # frameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    # frameWidth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    # frameHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    # if not frameCount:
    #     print 'Video Not Found.'
    #     exit(1)

    # if isGray:
    #     buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))
    # else:
    #     buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    # fc = 0
    # ret = True

    # while (fc < frameCount  and ret):
    #     try:
    #         ret, frame = cap.read()
    #         if isGray:
    #             frame = cv2.cvtColor(frame, cv2.cv.CV_CV_COLOR_BGR2GRAY)
    #         buf[fc] = frame

    #         fc += 1
    #     except(Exception):
    #         print 'Failed opening the video'
    #         raise


    # cap.release()
    # cv2.destroyAllWindows()

    # print 'frameCount:', frameCount
    # print 'frameHeight:', frameHeight
    # print 'frameWidth:', frameWidth
    # cv2.imshow('frame', buf[9])
    # cv2.waitKey(0)

    # read as uint8 array
    buf = skvideo.io.vread(filename, outputdict={"-pix_fmt": "gray"})
    if not isGray:
        buf = skvideo.io.vread(filename)
    return buf