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

def Radience_cal(image,A,Transmission_map,t_not):
    #Used information from the transmit map to remove haze from the image. 
    image_copy = image.copy()
    A_copy = A.copy()
    Transmission_map_copy = (Transmission_map.copy()).astype(float)
    divisor = np.maximum(Transmission_map_copy,t_not)
    radience = (image.copy()).astype(float)
    for i in range(3):
        radience[:,:,i] = np.divide(((image_copy[:,:,i]).astype(float) - A[0,0,i]),divisor) + A[0,0,i]
    #radience = 255*(radience/np.max(radience))
    radience[radience>255]=255
    radience[radience<0]=0
    return radience.astype('uint8')

def L_calculator(image,Transmission_map):
    #helps fine tune the transmition map for a better result
    epsalon = 10**(-8)
    h,w = image.shape[:2]
    window_area = (2*r + 1)**2
    n_vals = (w - 2*r)*(h - 2*r)*window_area**2
    k = 0
    # data for matting laplacian in coordinate form
    i = np.empty(n_vals, dtype=np.int32)
    j = np.empty(n_vals, dtype=np.int32)
    v = np.empty(n_vals, dtype=np.float64)

    # for each pixel of image
    for y in range(r, h - r):
        for x in range(r, w - r):

            # gather neighbors of current pixel in 3x3 window
            n = image[y-r:y+r+1, x-r:x+r+1]
            u = np.zeros(3)
            for p in range(3):
                u[p] = n[:, :, p].mean()
            c = n - u

            # calculate covariance matrix over color channels
            cov = np.zeros((3, 3))
            for p in range(3):
                for q in range(3):
                    cov[p, q] = np.mean(c[:, :, p]*c[:, :, q])

            # calculate inverse covariance of window
            inv_cov = np.linalg.inv(cov + epsilon/window_area * np.eye(3))

            # for each pair ((xi, yi), (xj, yj)) in a 3x3 window
            for dyi in range(2*r + 1):
                for dxi in range(2*r + 1):
                    for dyj in range(2*r + 1):
                        for dxj in range(2*r + 1):
                            i[k] = (x + dxi - r) + (y + dyi - r)*w
                            j[k] = (x + dxj - r) + (y + dyj - r)*w
                            temp = c[dyi, dxi].dot(inv_cov).dot(c[dyj, dxj])
                            v[k] = (1.0 if (i[k] == j[k]) else 0.0) - (1 + temp)/window_area
                            k += 1
    h,w = Transmission_map.shape
    L = scipy.sparse.csr_matrix((v, (i, j)), shape=(w*h, w*h))
    return L

def soft_matting(L,image,t_map):
    image_copy = image.copy()
    lamda = 10**(-4)
    U = np.identity(L.shape[0])
    t_map_mat = t_map*(L+lamda*U)/lamda
    return t_map_mat

def guided_filter(image,guide,diameter,epsilon):
    w_size = diameter+1
    # Exatrcation the mean of the image by blurring
    meanI=cv2.blur(image,(w_size,w_size))
    mean_Guide=cv2.blur(guide,(w_size,w_size))
    # Extracting the auto correlation
    II=image**2
    corrI=cv2.blur(II,(w_size,w_size))
    # Finding the correlation between image and guide
    I_guide=image*guide
    corrIG=cv2.blur(I_guide,(w_size,w_size))
    # using the mean of the image to find the variance of each point
    varI=corrI-meanI**2
    covIG=corrIG-meanI*mean_Guide
    #covIG normalized with a epsilon factor
    a=covIG/(varI+epsilon)
    #a is used to find the b 
    b=mean_Guide-a*meanI 
    meanA=cv2.blur(a,(w_size,w_size))
    meanB=cv2.blur(b,(w_size,w_size))
    transmission_rate=meanA*image+meanB
    # normalizaing of the transimational map
    transmission_rate = transmission_rate/np.max(transmission_rate)
    return transmission_rate



#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

# inputing the information
base_path  =  os.getcwd()
test_Haze = os.listdir(base_path+ '/data_set/Training_Set/hazy')
test_GT = os.listdir(base_path+ '/data_set/Training_Set/GT')
image = cv2.imread( base_path + "/data_set/Test_Set/Bridge.jpg",cv2.IMREAD_COLOR)
# extracting the minmum value from 15 by 15 patch
min_image = min_filter(image)

# perfroming the minmin with 15by15 min filter
dark_prior = rgb_min_image(min_image)
# displaying the results
fig, axes= plt.subplots(nrows=1, ncols=3,figsize=(20,5))
plt.suptitle('Stages of Dark channel')
axes[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
axes[0].set_title('original image')
axes[1].imshow(cv2.cvtColor(min_image,cv2.COLOR_BGR2RGB))
axes[1].set_title('The min 15 patch image',)
axes[2].imshow(dark_prior,cmap='gray')
axes[2].set_title('The dark prior')
interactive(True)
plt.show()

A = A_estimator(image,dark_prior)
fig, axes= plt.subplots(nrows=1, ncols=3,figsize=(20,5))
axes[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
axes[0].set_title('original image')
axes[1].imshow(A,cmap='gray')
axes[1].set_title('The Ambiance image')
Transmition_image = transmition_map(image,A,0.95)
axes[2].imshow(Transmition_image,cmap='gray')
axes[2].set_title('The transmitance image')

plt.show()

fig, axes= plt.subplots(nrows=1, ncols=3,figsize=(20,5))
radience_image = Radience_cal(image,A,Transmition_image,0.1)
axes[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
axes[0].set_title('original image')
axes[1].imshow(Transmition_image,cmap='gray')
axes[1].set_title('The transmitance image')
axes[2].imshow(cv2.cvtColor(radience_image,cv2.COLOR_BGR2RGB))
axes[2].set_title('Haze Free image')

plt.show()

epsilon = 10**-8
img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# refined the transmition map using the guide filter
refine_Transmission_image=guided_filter(img_gray.astype(np.float32),Transmition_image.astype(np.float32),100,epsilon)
refine_radience_image = Radience_cal(image,A,refine_Transmission_image,0.1)
# diplaying the refined results
fig, axes= plt.subplots(nrows=1, ncols=3,figsize=(20,5))
radience_image = Radience_cal(image,A,Transmition_image,0.1)
axes[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
axes[0].set_title('original image')
axes[1].imshow(refine_Transmission_image,cmap='gray')
axes[1].set_title('The Refine Transmitance image')
axes[2].imshow(cv2.cvtColor(refine_radience_image,cv2.COLOR_BGR2RGB))
axes[2].set_title('Haze Free image')
interactive(False)
plt.show()