from django.shortcuts import render
from PIL import Image
import numpy as np
import cv2
from .FaceDetect import FaceCoordinates
from .SiameseNN import SiameseNetwork
from .ConvertTensor import ConvertTensor

def homepage(request):
    f1 = FaceCoordinates()
    net = SiameseNetwork()
    # form = ImageForm(request.POST , request.FILES)
    if request.method == 'POST':
        img1 , img2 = Image.open(request.FILES['file1']) , Image.open(request.FILES['file2'])
        data1 , data2 = np.array(img1) , np.array(img2)
        location1 , location2 = f1.FaceRecognize(data1 , data2)
        location1 , location2 =  ConvertTensor().Convert(location1 , location2)
        pred = net(location1 , location2)
        print(pred)
    return render(request , 'Root/homepage.html')