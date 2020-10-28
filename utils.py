import numpy as np
from os import getcwd, path
from cv2 import imread, IMREAD_COLOR

def read_image(path=None):
    '''
        Read an image from a path
        Path is relative path or full path
    '''
    base_path = getcwd()
    full_path = path.join(base_path,path)
    if base_path in path:
        full_path = path
    
    if not(path.exists(full_path)):
        print('The path \" {}\"does not exist. Make just that the file exist').fromat(full_path)
        return None
    else:
        image = imread(full_path,IMREAD_COLOR)
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

def soft_matting(L,image,t_map):
    image_copy = image.copy()
    lamda = 10**(-4)
    U = np.identity(L.shape[0])
    t_map_mat = t_map*(L+lamda*U)/lamda
    return t_map_mat