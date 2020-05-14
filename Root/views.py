from django.shortcuts import render
from PIL import Image
import numpy as np
import cv2
from .FaceDetect import FaceCoordinates
from .NN import SiameseNetwork
from .ProcessImage import ProcessImage
import torch.nn.functional as F
import torch

def homepage(request):
    f1 = FaceCoordinates()
    net = SiameseNetwork()
    similarity = 0
    if request.method == 'POST':
        img1 = Image.open(request.FILES['file1']) # PIL Image
        img2 = Image.open(request.FILES['file2']) # PIL Image

        location1 , location2 = f1.FaceRecognize(img1 , img2)

        location1 = ProcessImage().Process(location1)
        location2 = ProcessImage().Process(location2)

        output1 , output2 = net(torch.from_numpy(location1) , torch.from_numpy(location2))
        similarity = F.pairwise_distance(output1 , output2).item()
        
    return render(request , 'Root/homepage.html' , {'similarity' : float(round(float(similarity) , 2)) })