import cv2 

class FaceCoordinates:

    def __init__(self):
        pass

    def GetLocation(self , image_array):
        image = image_array
        if len(image.shape) !=  2:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
                                image,
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

    

