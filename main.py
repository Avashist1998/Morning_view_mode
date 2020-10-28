import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib import interactive
import scipy


def read_image(path=None):
    '''
        Read an image from a path
        Path is relative path or full path
    '''
    base_path = os.getcwd()
    full_path = os.path.join(base_path,path)
    if base_path in path:
        full_path = path
    
    if not(os.path.exists(full_path)):
        print('The path \" {}\"does not exist. Make just that the file exist').fromat(full_path)
        return None
    else:
        image = cv2.imread(full_path,cv2.IMREAD_COLOR)
        return image

def rgb_min_image(image):
    # extractes the min of the rgb values and outputs
    # a gray scale image
    rgb_image = np.amin(image, axis= 2)
    return rgb_image

def min_filter(image):
    # perfroms the min filter on 15 by 15 area
    for k in range (3):
        # creating a copy of the filter 
        i_image = image.copy()
        # extracting one channel of the image 
        temp_image = image[:,:,k].copy()
        [row,col] = temp_image.shape
        # padding the iamge 
        temp_image = cv2.copyMakeBorder(temp_image, 14, 14, 14, 14, cv2.BORDER_REFLECT) 
        # perfroming the min filter with 15 x 15 window
        for i in range(row):
            for j in range(col):
                i_image[i,j,k] = (temp_image[i:15+i,j:15+j]).min()
    return i_image

def dark_channel(image):
    # output the dark channel as the image
    new_image = image.copy()
    # perfroming the 15 x 15 min filter 
    min_image = min_filter(new_image)
    # perfroming the color min operation
    dark_prior = rgb_min_image(min_image)
    return dark_prior

def transmission_map(image,A,w):
    #finds the transmission map for the image
    image_new =  np.divide(image,A).astype(float)
    # finding the dark channel of the divide image 
    new_dark = dark_channel(image_new)
    # Saling and subtracting the image 
    transmission  = 1 - w*new_dark
    return transmission 

def A_estimator(image,dark_prior):
    #Used the information extracted from the dark prior 
    #find a value for A 
    image_copy = image.copy()
    [row,col,dem] = image_copy.shape
    dark_copy = dark_prior.copy()
    # finding the number of 0.01% values
    num = np.round(row*col*0.001).astype(int)
    j = sorted(np.asarray(dark_copy).reshape(-1), reverse=True)[:num]
    # getting the location of the top 0.01%
    ind = np.unravel_index(j[0], dark_copy.shape)
    # Pefroming a search for the max value in the group
    max_val = image_copy[ind[0],ind[1],:]
    for element in j:
        ind = np.unravel_index(element, dark_copy.shape)
        if (sum(max_val[:]) < sum(image_copy[ind[0],ind[1],:])):
            max_val[:] = image_copy[ind[0],ind[1],:]
    # creating a color image of the max value
    A = image_copy
    A[:,:,:] = max_val[:]
    return A

def Radience_cal(image,A,Transmission_map,t_not):
    #Used information from the transmit map to remove haze from the image. 
    image_copy = image.copy()
    Transmission_map_copy = (Transmission_map.copy()).astype(float)
    # Pefroming the min operation between Ttransmission map and 0.1
    divisor = np.maximum(Transmission_map_copy,t_not)
    radience = (image.copy()).astype(float)
    # Perfroming the eqution 3 for every color channel
    for i in range(3):
        radience[:,:,i] = np.divide(((image_copy[:,:,i]).astype(float) - A[0,0,i]),divisor) + A[0,0,i]
    # Capping all of the out of bound values 
    #radience = radience - np.min(radience)
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
    # using the mean of a b to fix refine the transmission map
    transmission_rate=meanA*image+meanB
    # normalizaing of the transimational map
    transmission_rate = transmission_rate/np.max(transmission_rate)
    return transmission_rate


def Haze_Remover(path=None, image=None,epsilon=10**-8):
    '''
    This function is used to dehaze a image from an image path or from a cv2 image oject
    '''
    if path is None and image is None:
        print("There is not path and image enter to the function. Please add a image or a path to the model")
        return None
    else:
        if image is none:
            image = read_image(path)
            min_image = min_filter(image)
            dark_prior = rgb_min_image(min_image)
            A = A_estimator(image,dark_prior)
            img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            Transmition_image = transmission_map(image,A,0.95)
            refine_Transmission_image = guided_filter(img_gray.astype(np.float32),Transmition_image.astype(np.float32),100,epsilon)
            refine_radience_image = Radience_cal(image,A,refine_Transmission_image,0.1)
            output = {'Input':image, 'Min_Image':min_image, 'A':A_estimator,'Gray_Image':img_gray,
                        'Transmition_Map':Transmition_image, 'Refine_Transmition_Map':refine_Transmission_image,
                        'DeHaze_Image':refine_radience_image}

    
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

# inputing the information
base_path  =  os.getcwd()
image = cv2.imread( base_path + "/data_set/Test_Set/Bridge.jpg",cv2.IMREAD_COLOR)
# extracting the minmum value from 15 by 15 patch
min_image = min_filter(image)

# perfroming the minmin with 15 by 15 min filter
dark_prior = rgb_min_image(min_image)
# displaying the results
A = A_estimator(image,dark_prior)
epsilon = 10**-8
img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
Transmition_image = transmission_map(image,A,0.95)
refine_Transmission_image=guided_filter(img_gray.astype(np.float32),Transmition_image.astype(np.float32),100,epsilon)
refine_radience_image = Radience_cal(image,A,refine_Transmission_image,0.1)

fig, axes= plt.subplots(nrows=1, ncols=3,figsize=(18,8))
plt.suptitle('Dark Channel Prior Stage')
axes[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
axes[0].set_title('Original image')
axes[1].imshow(cv2.cvtColor(min_image,cv2.COLOR_BGR2RGB))
axes[1].set_title('The min 15 patch image',)

axes[2].imshow(dark_prior,cmap='gray')
axes[2].set_title('The dark prior')
interactive(True)
plt.show()

A = A_estimator(image,dark_prior)
fig, axes= plt.subplots(nrows=3, ncols=1,figsize=(20,5))
axes[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
axes[0].set_title('original image')
axes[1].imshow(A,cmap='gray')
axes[1].set_title('The Ambiance image')
Transmition_image = transmission_map(image,A,0.95)
axes[2].imshow(Transmition_image,cmap='gray')
axes[2].set_title('The Transmission  map')

plt.show()

fig, axes= plt.subplots(nrows=1, ncols=3,figsize=(20,5))
radience_image = Radience_cal(image,A,Transmition_image,0.1)
axes[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
axes[0].set_title('original image')
axes[1].imshow(Transmition_image,cmap='gray')
axes[1].set_title('The Transmission  map')
axes[2].imshow(cv2.cvtColor(radience_image,cv2.COLOR_BGR2RGB))
axes[2].set_title('Haze Free image')

plt.show()

epsilon = 10**-8
img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# refined the transmition map using the guide filter
refine_Transmission_image=guided_filter(img_gray.astype(np.float32),Transmition_image.astype(np.float32),100,epsilon)
refine_radience_image = Radience_cal(image,A,refine_Transmission_image,0.1)
# diplaying the refined results
fig, axes= plt.subplots(nrows=2, ncols=1,figsize=(5,20))
radience_image = Radience_cal(image,A,Transmition_image,0.1)
axes[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
axes[0].set_title('original image')
axes[1].imshow(refine_Transmission_image,cmap='gray')
axes[1].set_title('The Refine Transmitance image')
axes[0].imshow(cv2.cvtColor(radience_image,cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Haze free image')

axes[1].imshow(cv2.cvtColor(refine_radience_image,cv2.COLOR_BGR2RGB))
axes[1].set_title('Refined Haze Free image')

interactive(False)
plt.show()