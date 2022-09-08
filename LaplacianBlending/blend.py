import cv2 as cv
from scipy import ndimage
import numpy as np
import imageio

# 5x5 Gaussian blur kernel
kernel = (1.0/256)*np.array(
  [[1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, 36, 24, 6], 
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1]])

#Upsamples the image at the rate of 2
def upsample(image):
  image_up = np.zeros((2*image.shape[0], 2*image.shape[1]))
  image_up[::2, ::2] = image
  # When the width and length are doubled, the image scales 4 times its original size so the kernel is multiplied by 4 for convolution
  return ndimage.filters.convolve(image_up, 4*kernel , mode='constant')

#Downsamples the image at the rate of 2            
def downsample(image):
  image_blur = ndimage.filters.convolve(image, kernel , mode='constant')
  return image_blur[::2, ::2]                                

#Method to generate both Gaussian and Laplacian pyramids of the image                                         
def get_pyramids(image):
  # Initializing the pyramids
  G = [image, ]
  L = []

  # Build the Gaussian pyramid
  while image.shape[0] >= 2 and image.shape[1] >= 2:
    image = downsample(image)
    G.append(image)

  # Build the Laplacian pyramid
  for i in range(len(G) - 1):
    #difference between Gaussian pyramid at level i and upsampled Gaussian pyramid at level i+1
    L.append(G[i] - upsample(G[i + 1]))
  return G[:-1], L

#Forms combined pyramid from LA and LB using nodes of GR as weights
def blend_pyramids(A, B, mask):
  [GA, LA] = get_pyramids(A)
  [GB ,LB] = get_pyramids(B)
  # Build a Gaussian pyramid GR from selected region R 
  [GMask, LMask] = get_pyramids(mask)
  # Form a combined pyramid LS using the formula LS(i,j) = GR(I,j)*LA(I, j) + (1-GR(I, j)* LB(I, j))
  blend = []
  for i in range(len(LA)):
    LS = (GMask[i]/255)*LA[i] + (1 - GMask[i]/255)*LB[i]
    blend.append(LS)
  return blend

#Reconstructs the pyramids
def reconstruct(pyramid):
  rows, cols = pyramid[0].shape
  res = np.zeros((rows, cols + cols//2), dtype= np.double)
  revPyramid = pyramid[::-1]
  stack = revPyramid[0]
  for i in range(1, len(revPyramid)):
    stack = upsample(stack) + revPyramid[i] 
  return stack


def laplacian_blending(img1, img2, mask):
  # split to 3 basic color
  img1R,img1G,img1B = cv.split(img1)
  img2R,img2G,img2B = cv.split(img2)
  #Apply laplacian blending and reconstruct the image
  R = reconstruct(blend_pyramids(img1R, img2R, mask))
  G = reconstruct(blend_pyramids(img1G, img2G, mask))
  B = reconstruct(blend_pyramids(img1B, img2B, mask))

  #Merges the RGB layers
  output = cv.merge((B, G, R))
  imageio.imsave('output.png',output)
  img = cv.imread('output.png')
  cv.imshow('Result',img)
  cv.waitKey(0)
  cv.destroyAllWindows()

#Read the images and mask
apple = cv.imread('apple.jpg')
orange = cv.imread('orange.jpg')
#Read the mask as a grayscale image
mask = cv.imread('mask.jpg', 0)

#Function call to perform laplacian blending
laplacian_blending(apple, orange, mask)