import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib import interactive
import scipy

def rgb_min_image(image):
    # extractes the min of the rgb values and outputs
    # a gray scale image
    rgb_image = np.amin(image, axis= 2)
    return rgb_image

def min_filter(image):
    # perfroms the min filter on 15 by 15 area
    for k in range (3):
        i_image = image.copy()
        temp_image = image[:,:,k].copy()
        [row,col] = temp_image.shape
        temp_image = cv2.copyMakeBorder(temp_image, 14, 14, 14, 14, cv2.BORDER_REFLECT) 
        for i in range(row):
            for j in range(col):
                i_image[i,j,k] = (temp_image[i:14+i,j:14+j]).min()
    return i_image

def dark_channel(image):
    # output the dark channel as the image
    new_image = image.copy()
    min_image = min_filter(new_image)
    dark_prior = rgb_min_image(min_image)
    return dark_prior

def transmition_map(image,A,w):
    #finds the transmition map for the image
    image_new =  np.divide(image,A).astype(float)
    new_dark = dark_channel(image_new)
    transmition = 1 - w*new_dark
    return transmition

def A_estimator(image,dark_prior):
    #Used the information extracted from the dark prior 
    #find a value for A 
    image_copy = image.copy()
    [row,col,dem] = image_copy.shape
    dark_copy = dark_prior.copy()
    num = np.round(row*col*0.001).astype(int)
    j = sorted(np.asarray(dark_copy).reshape(-1), reverse=True)[:num]
    ind = np.unravel_index(j[0], dark_copy.shape)
    max_val = image_copy[ind[0],ind[1],:]
    for element in j:
        ind = np.unravel_index(element, dark_copy.shape)
        if (sum(max_val[:]) < sum(image_copy[ind[0],ind[1],:])):
            max_val[:] = image_copy[ind[0],ind[1],:]
    A = image_copy
    A[:,:,:] = max_val[:]
    return A

def Radience_cal(image,A,transmit_map,t_not):
    #Used information from the transmit map to remove haze from the image. 
    image_copy = image.copy()
    A_copy = A.copy()
    transmit_map_copy = (transmit_map.copy()).astype(float)
    divisor = np.maximum(transmit_map_copy,t_not)
    radience = (image.copy()).astype(float)
    for i in range(3):
        radience[:,:,i] = np.divide(((image_copy[:,:,i]).astype(float) - A[0,0,i])/255,divisor) + A[0,0,i]/255
    radience = (((radience/np.max(radience)))*255).astype('uint8')
    return radience

def L_calculator(image):
    #helps fine tune the transmition map for a better result
    epsalon = 10**(-8)
    [row, col, dem] = image.shape
    abs_wk = 9
    num_pixels = row*col
    L = scipy.sparse.csr_matrix((num_pixels,num_pixels),dtype= 'float')
    #padding wiht the image 
    L = cv2.copyMakeBorder(L[:,:], 1, 1, 1, 1, cv2.BORDER_REFLECT)
    for i in range(1,row-1):
        for j in range(1,col-1):
            w = image[i-1:i+1,j-1:j+1,:]
            w_col = w.reshape(abs_wk,dem)
            w_mean = np.mean(w).T
            w_conv = np.cov(w,1)
            w_inv_conv_idetity = w_conv + (epsalon/abs_wk) * scipy.sparce.speye(3,3)
            
            L[i,j] = 1
    print('test')
def soft_matting(L,image,t_map):
    image_copy = image.copy()
    lamda = 10**(-4)
    U = np.identity(L.shape[0])
    t_map_mat = t_map*(L+lamda*U)/lamda
    return t_map_mat

# inputing the information
base_path  =  os.getcwd()
test_Haze = os.listdir(base_path+ '/data_set/Training_Set/hazy')
test_GT = os.listdir(base_path+ '/data_set/Training_Set/GT')
image = cv2.imread( base_path + "/data_set/Test_Set/Golden_gate.jpg",cv2.IMREAD_COLOR)
# extracting the minmum value from 15 by 15 patch
min_image = min_filter(image)

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
ax3.set_title('Haze Free image')
interactive(False)
plt.show()