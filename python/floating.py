import cv2
import numpy as np
from face_common import get_landmarks, get_landmarks_points


def floating_face(models, image, background):

    face, landmarks = get_landmarks(models, image)
    lps = get_landmarks_points(landmarks)
    np_points = np.array(lps, np.int32)
    hull = cv2.convexHull(np_points)

    #mask = np.zeros((image.shape[1], frame.shape[0]), dtype='uint8')
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [hull], 0, color=(255, 255, 255), thickness=-1)

    mask = cv2.GaussianBlur(mask, ksize=(19, 19), sigmaX=16)
    mask = mask.astype(float) / 255
    mask_inv = 1 - mask
    ret = (image * mask + background * mask_inv).astype(dtype='uint8')

    return ret