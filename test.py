import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib import interactive
import scipy
def rgb_min_image(image):
    rgb_image = np.amin(image, axis= 2)
    return rgb_image

def dark_channel(image):
    for k in range (3):
        i_image = image.copy()
        temp_image = image[:,:,k].copy()
        [row,col] = temp_image.shape
        temp_image = cv2.copyMakeBorder(temp_image, 14, 14, 14, 14, cv2.BORDER_REFLECT) 
        for i in range(row):
            for j in range(col):
                i_image[i,j,k] = (temp_image[i:14+i,j:14+j]).min()
    return i_image

def transmition_map(image,A,w):
    image_copy = ((image/255).astype(float)).copy()
    A_copy = ((A/255).astype(float)).copy()
    [row,col,dem] = image_copy.shape
    image_copy = np.divide(image_copy,A_copy)
    image_r = cv2.copyMakeBorder(image_copy[:,:,0], 14, 14, 14, 14, cv2.BORDER_REFLECT)
    image_g = cv2.copyMakeBorder(image_copy[:,:,1], 14, 14, 14, 14, cv2.BORDER_REFLECT)
    image_b = cv2.copyMakeBorder(image_copy[:,:,2], 14, 14, 14, 14, cv2.BORDER_REFLECT)
    i_image = image_copy
    for i in range(row):
        for j in range(col):
            i_image[i,j,0] = (image_r[i:14+i,j:14+j]).min()
            i_image[i,j,1] = (image_g[i:14+i,j:14+j]).min()
            i_image[i,j,2] = (image_b[i:14+i,j:14+j]).min()
    i_image = np.amin(i_image, axis= 2)
    transmition = 1 - w*(i_image)
    transmition = (transmition*255).astype(int)
    return transmition

def A_estimator(image,dark_prior):
    image_copy = image.copy()
    [row,col,dem] = image_copy.shape
    dark_copy = dark_prior.copy()
    num = np.round(row*col*0.01).astype(int)
    j = sorted(np.asarray(dark_copy).reshape(-1), reverse=True)[:num]
    ind = np.unravel_index(j[0], dark_copy.shape)
    max_val = image_copy[ind[0],ind[1],:]
    for element in j:
        ind = np.unravel_index(element, dark_copy.shape)
        if (sum(max_val[:])/3 > sum(image_copy[ind[0],ind[1],:])/3):
            max_val[:] = image_copy[ind[0],ind[1],:]
    A = image_copy
    A[:,:,:] = max_val[:]
    return A

def Radience_cal(image,A,transmit_map,t_not):
    image_copy = image.copy()
    A_copy = A.copy()
    transmit_map_copy = (transmit_map.copy()).astype(float)
    transmit_map_copy = transmit_map_copy/255
    divisor = np.maximum(transmit_map_copy,t_not)
    radience = (image.copy()).astype(float)
    for i in range(3):
        radience[:,:,i] = np.divide(((image_copy[:,:,i]).astype(float) - A[0,0,i])/255,divisor) + A[0,0,i]/255
    radience = (((radience/np.max(radience)))*255).astype('uint8')
    return radience
def L_calculator(image,):
    print('test')
def soft_matting(image,lamda,t_map):
    
    print(image)

# inputing the information
base_path  =  os.getcwd()
test_Haze = os.listdir(base_path+ '/data_set/Training_Set/hazy')
test_GT = os.listdir(base_path+ '/data_set/Training_Set/GT')
image = cv2.imread( base_path + "/data_set/Test_Set/Iran.jpg",cv2.IMREAD_COLOR)
# extracting the minmum value from 15 by 15 patch
min_image = dark_channel(image)

# perfroming the minmin with 15by15 min filter
dark_prior = rgb_min_image(min_image)
# displaying the results
plt.fig, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
ax1.set_title('original image')
ax2.imshow(cv2.cvtColor(min_image,cv2.COLOR_BGR2RGB))
ax2.set_title('The min 15 patch image',)
ax3.imshow(dark_prior,cmap='gray')
ax3.set_title('The dark prior')
interactive(True)
plt.show()

A = A_estimator(image,dark_prior)
plt.fig, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
ax1.set_title('original image')
ax2.imshow(A,cmap='gray')
ax2.set_title('The Ambiance image')
Transmit_image = transmition_map(image,A,0.95)
ax3.imshow(Transmit_image,cmap='gray')
ax3.set_title('The transmitance image')

plt.show()

plt.fig, (ax1,ax2,ax3) = plt.subplots(1,3)
radience_image = Radience_cal(image,A,Transmit_image,0.1)
ax1.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
ax1.set_title('original image')
ax2.imshow(Transmit_image,cmap='gray')
ax2.set_title('The transmitance image')
ax3.imshow(cv2.cvtColor(radience_image,cv2.COLOR_BGR2RGB))
ax3.set_title('radiance image')
interactive(False)
plt.show()