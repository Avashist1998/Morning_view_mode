import cv2
import numpy as np
from utils import read_image, rgb_min_image, min_filter

class DarkPrior():
    def __init__(self, epsilon=10**-8):
        self.epsilon = epsilon


    def dark_channel(self, image):
        # output the dark channel as the image
        new_image = image.copy()
        # perfroming the 15 x 15 min filter 
        min_image = min_filter(new_image)
        # perfroming the color min operation
        dark_prior = rgb_min_image(min_image)
        return dark_prior

    def transmission_map(self, image,A,w):
        #finds the transmission map for the image
        image_new =  np.divide(image,A).astype(float)
        # finding the dark channel of the divide image 
        new_dark = self.dark_channel(image_new)
        # Saling and subtracting the image 
        transmission  = 1 - w*new_dark
        return transmission 

    def A_estimator(self, image,dark_prior):
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

    def Radience_cal(self, image,A,Transmission_map,t_not):
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

    def guided_filter(self, image,guide,diameter,epsilon):
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

    def Haze_Remover(path=None, image=None):
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
            Transmition_image = self.transmission_map(image,A,0.95)
            refine_Transmission_image = self.guided_filter(img_gray.astype(np.float32),Transmition_image.astype(np.float32),100,self.epsilon)
            refine_radience_image = self.Radience_cal(image,A,refine_Transmission_image,0.1)
            self.output = {'Input':image, 'Min_Image':min_image, 'A':A_estimator,'Gray_Image':img_gray,
                        'Transmition_Map':Transmition_image, 'Refine_Transmition_Map':refine_Transmission_image
                        'DeHaze_Image':refine_radience_image}
            return output

    def Save_image(self, path='output.jpg', key='DeHaze_Image'):
        
