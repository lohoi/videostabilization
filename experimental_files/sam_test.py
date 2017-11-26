#!/usr/bin/env python
import cv2

if __name__ == "__main__":
    try:
        cap = cv2.VideoCapture("../media/test_vid_eric.mp4")
    except:
        print "Failed opening the video"
        raise

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break;
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
