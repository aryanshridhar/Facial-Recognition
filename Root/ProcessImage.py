from PIL import Image
import numpy as np 
import torch

class ProcessImage:
    def __init__(self):
        pass
    
    @staticmethod
    def Process(image_array):
        img = np.moveaxis(image_array , -1 ,0)[1,:,:]
        img = np.expand_dims(img , 0)
        img = np.expand_dims(img , 0)

        return img
