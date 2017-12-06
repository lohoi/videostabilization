import numpy as np
import cv2
import matplotlib.pyplot as plt

def play_video(filename):
    '''Plays video.'''
    pass

def plot_path(C_):
    '''Plot estimated path'''
    point = np.array([[0],[0],[1]])
    frames = np.arange(len(C_) + 1)
    frame_cum = np.eye(3)
    X = [0]
    Y = [0]
    for transform in C_:
        frame_cum = np.dot(frame_cum, transform)
        tep_point = np.dot(frame_cum, point)
        X.append(temp_point[0,0])
        Y.append(temp_point[1,0])
    plt.subplot(1,2,1)
    plt.plot(frames,X,'r--')
    plt.subplot(1,2,2)
    plt.plot(frames,Y,'r--')
    plt.show()

def plot_new_path(C_,B_):
    '''Plot estimated path'''
    point = np.array([[0],[0],[1]])
    frames = np.arange(len(C_))
    frame_cum = np.eye(3)
    old_X = []
    old_Y = []
    new_X = []
    new_Y = []
    for ind, transform in enumerate(C_):
        frame_cum = np.dot(frame_cum, transform)
        temp_point = np.dot(frame_cum, point)
        old_X.append(temp_point[0,0])
        old_Y.append(temp_point[1,0])
        temp = np.dot(np.dot(frame_cum, B_[ind]),point)
        new_X.append(temp[0,0])
        new_Y.append(temp[1,0])
    plt.subplot(1,2,1)
    plt.plot(frames,old_X,'r--',frames,new_X,'b--')
    plt.subplot(1,2,2)
    plt.plot(frames,old_Y,'r--',frames,new_Y,'b--')
    plt.show()

def draw_matches(img1, kp1, img2, kp2, matches):
    """
    source: https://stackoverflow.com/questions/20259025/
            module-object-has-no-attribute-drawmatches-opencv-python

    Used in this project to view the output of SIFT descriptors in
    estimate_path.py. We found that L2 norm produces less noisy matches
    contrary to David Lowe's nearest-neighbor ratio-test as described
    in his SIFT paper.

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    # Create the output image
    # The rows of the output are the largest between the two images
    # and the columns are simply the sum of the two together
    # The intent is to make this a colour image, so make this 3 channels
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out
