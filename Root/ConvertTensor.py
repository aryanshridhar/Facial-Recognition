import torch
import numpy as np

class ConvertTensor:
    def __init__(self):
        pass

    @staticmethod
    def Convert(img1 , img2):
        img1 , img2  = np.moveaxis(img1 , -1 ,0) , np.moveaxis(img2 , -1 , 0)
        print(img1.shape)
        tensor1 , tensor2 = torch.from_numpy(img1) , torch.from_numpy(img2)
        tensor1 = tensor1[:,:,0]
        tensor2 = tensor2[:,:,0]
        print(tensor1.shape)
        print(tensor2.shape)

        return tensor1 , tensor2
