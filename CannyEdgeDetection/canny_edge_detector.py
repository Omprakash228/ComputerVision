import cv2 as cv
import numpy as np
from scipy.ndimage.filters import convolve

def gradient_estimation(image, ksize, sigma):
    #generating Gaussian kernel
    size = int(ksize) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal_constant = 1 / (2.0 * np.pi * sigma**2)

    #Computing using the 2D gaussian kernel equation
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal_constant

    #smooth the image
    smooth_img = convolve(image, g, mode='constant')

    #compute magnitude and orientation fo the gradients
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

    Ix = convolve(smooth_img, Kx, mode='constant')
    Iy = convolve(smooth_img, Ky, mode='constant')

    G = np.sqrt(np.square(Ix) + np.square(Iy))
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)
   

def non_max_suppression(mag, theta):
    r, c = mag.shape
    output = np.zeros(mag.shape)
    
    theta = theta * 180 / np.pi
    #adding 180 to the angles that are less than 0
    theta[theta < 0] += 180

    #iterate through the image and find before and after pixel based on the edge direction
    for i in range(0, r - 1):
        for j in range(0, c - 1):
            angle = theta[i, j]
            before = 255
            after = 255

            if (0 <= angle < np.pi/8) or (7*np.pi/8 <= angle <= np.pi):
                before = mag[i, j-1]
                after = mag[i, j+1]

            elif (np.pi/8 <= angle < 3*np.pi/8):
                before = mag[i-1, j+1]
                after = mag[i+1, j-1]

            elif (3*np.pi/8 <= angle < 5*np.pi/8):
                before = mag[i-1, j]
                after = mag[i+1, j]
            
            elif (5*np.pi/8 <= angle < 7*np.pi/8):
                before = mag[i+1, j+1]
                after = mag[i-1, j-1]
            
            # if current pixel is more intense then the value is kept, else it is set to 0
            if(mag[i, j] >= before and mag[i,j] >= after):
                output[i,j] = mag[i,j]
            else:
                output[i,j] = 0
    
    return output

def threshold_hysteresis(img, low, high, low_pixel, high_pixel):
    h_threshold = img.max() * high
    l_threshold = h_threshold * low

    r, c = img.shape
    output = np.zeros(img.shape)

    weak = low_pixel
    strong = high_pixel

    strong_r, strong_c = np.where(img >= h_threshold)
    zeros_r, zeros_c = np.where(img < l_threshold)
    weak_r, weak_c = np.where((img <= h_threshold) & (img >= l_threshold))

    #setting pixel values based on the high and low threshold values
    output[strong_r, strong_c] = strong
    output[weak_r, weak_c] = weak
    output[zeros_r, zeros_c] = 0

    #transforming weak pixels into strong, if atlease one of the surrounding pixels is strong
    for i in range(1, r-1):
        for j in range(1, c-1):
            if(output[i, j] == weak):
                if(output[i+1, j-1] == strong) or (output[i+1, j] == strong) or (output[i+1, j+1] == strong) or (output[i, j-1] == strong) or (output[i, j+1] == strong) or (output[i-1, j-1] == strong) or (output[i-1, j] == strong) or (output[i-1, j+1] == strong):
                    output[i,j] = strong
                else:
                    output[i,j] = 0
                
    return output

def CannyEdgeDetection(image, low_threshold, high_threshold):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mag, theta = gradient_estimation(img, 5, 1.5)
    non_max = non_max_suppression(mag, theta)
    img_final = threshold_hysteresis(non_max, 0.69, 0.7, low_threshold, high_threshold)

    return img_final

# images = [cv.imread('lena.png'), cv.imread('bird.png'), cv.imread('cameraman.jpg')]
# img = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)

# #computing gradient estimation after gaussian blue of size 5x5 and sigma=3.5
# mag, theta = gradient_estimation(img, 5, 3.5)
# non_max = non_max_suppression(mag, theta)
# #hysteresis thresholding with low_threshold = 0.05 and high_threshold = 0.95
# img_final = threshold_hysteresis(non_max, 0.05, 0.95)

# cv.imshow('final', img_final)
# cv.waitKey(0)
# cv.destroyAllWindows()


