import cv2 
import numpy as np

class FaceCoordinates:

    def __init__(self):
        pass

    def GetLocation(self , image): # Takes in PIL Image
        image = np.array(image) # Convert to Numpy array
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
                                gray,
                                scaleFactor=1.3,
                                minNeighbors=3,
                                minSize=(50, 50)
                            )
        for (x, y, w, h) in faces:
            image = image[y:y + h, x:x + w]
            
        return image 

    def FaceRecognize(self , img1 , img2):
        f_img1 = self.GetLocation(img1)
        f_img2 = self.GetLocation(img2)

        return f_img1 , f_img2

    

