import sys
from scipy.linalg import norm
from scipy import sum, average
import cv2
def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr
def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng
def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    if(len(img1)<=0 or len(img2)<=0):
        return(9999999,5555)
    if(img1.shape[0]>=64 and img1.shape[1]>=64 and img2.shape[0]>=64 and img2.shape[1]>=64):    
        img1 = cv2.resize(img1,(64,64),cv2.INTER_AREA)
        img2 = cv2.resize(img2,(64,64),cv2.INTER_AREA)
        # print(img1.shape,", ",img2.shape,"shape1")
        img1 = to_grayscale(img1)
        img2 = to_grayscale(img2)
        img1 = normalize(img1)
        img2 = normalize(img2)
        # calculate the difference and its norms
        diff = img1 - img2  # elementwise for scipy arrays
        m_norm = sum(abs(diff))  # Manhattan norm
        z_norm = norm(diff.ravel(), 0)  # Zero norm
        # print(m_norm,", ",z_norm,"checkmenorm")
        return (m_norm, z_norm)
    else:
        return(9999999,5555)