#!/usr/bin/env python

import cv2
import numpy as np

if __name__ == "__main__":
    cap = cv2.VideoCapture("../media/test_vid_eric.mp4")
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        print("Mission: ", ret)
        if not ret:
            print("End of video.")
            break;

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
