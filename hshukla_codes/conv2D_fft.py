# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 22:38:23 2021

@author: hritwik
"""

import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt


img = cv.imread('lena.png')
#img = cv.imread('wolves.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sz = img.shape
img_original = img
L=2
f = img

#F = np.zeros((sz[0],sz[1]), dtype="complex_")
g = np.zeros((sz[0],sz[1]), dtype="complex_")


#Question 2 (a)
#DFT2 function defination
def DFT2(f):
    
    im_flat = f.flatten()
    
    f_min = min(im_flat)
    
    f_max = max(im_flat)
    d_max = f_max-f_min
    
    print ("MIN = ", f_min)
    print ("MAX = ", f_max)
    print ("Difference max", d_max)
    # Histogram Equalization
    
    im_eq = np.zeros((sz[0],sz[1]))
    
    for i in range(sz[0]):
        for j in range(sz[1]):
            
            im_eq[i,j] = (((L-1)*(f[i,j]-f_min))/d_max)
   
    img = im_eq
 
    f_1 = np.zeros((sz[0],sz[1]), dtype="complex_")
    f_2 = np.zeros((sz[0],sz[1]), dtype="complex_")
   
    for i in range(sz[0]):
        f_1[i,:] = np.fft.fft(img[i,:])
    
    for j in range(sz[1]):    
        f_2[:,j] = np.fft.fft(f_1[:,j])

    F = f_2

    return F
 
#Question 2 (b)
#IDFT2 function defination
   
def IDFT2(F):    
    #Shift the origin to centre
    fshift = np.fft.fftshift(F)
    
    #Visualize MAGNITUDE SPECTRUM
    magnitude_spectrum = 25*np.log(1+np.abs(fshift))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype = np.uint8)
    cv.imshow("magnitude spectrum", magnitude_spectrum)
    
    #Visualize PHASE SPECTRUM
    phase_spectrum = np.angle(fshift)
    cv.imshow("Phase", phase_spectrum)

    #Again re-shift the origin to top left corner of the image
    inv_fshift = np.fft.ifftshift(fshift)
    
    f_3 = np.zeros((sz[0],sz[1]), dtype="complex_")
    f_4 = np.zeros((sz[0],sz[1]), dtype="complex_")
    
    #Finding 2D inverse of F using 1D inbuilt function
    for i in range(sz[1]):
        f_3[:,i] = np.fft.ifft(inv_fshift[:,i])
    
    
    for j in range(sz[0]):    
        f_4[j,:] = np.fft.ifft(f_3[j,:])
    
    g = f_4
    
    #Find the Difference in input image and inverse of 2D FFT image
    dif1 = g.real - img_original
    
    # Display difference in input image and inverse of 2D FFT image
    # Difference should be a black image as it is zero everywhere
    cv.imshow("difference in original image (f) and inverse (g) ", dif1)
    
    #Display final image g = IDFT2(F)
    cv.imshow("Final Image, g = IDFT2(F)", g.real)
    
    
    return g
    

#Calling the DFT2 function for 2-D FFT 
F = DFT2(f)

#Calling IDFT Function  for finding the inverse of FFT
g1 = IDFT2(F)

cv.waitKey(0)
cv.destroyAllWindows()
    