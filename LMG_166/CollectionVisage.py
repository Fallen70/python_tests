import cv2
import numpy
import time
import os

TRAINSET = "/usr/share/opencv/lbpcascades/lbpcascade_frontalface.xml"
DOWNSCALE = 4
IMAGE_SIZE = 170
NUMBER_OF_CAPTURE = 10

class Collection():

    def __init__(self, images_path, interval = 0.1 ):
        self.images_path = images_path
        self.camera = None
        self.classifier = None
        self.interval = interval

    def prepare_camera(self, capture_device):
        self.camera = cv2.VideoCapture( capture_device )
        self.classifier = cv2.CascadeClassifier( TRAINSET )

    def is_camera_available(self):
        if self.camera.isOpened():
            rval, frame = self.camera.read()
            return rval
        else:
            rval = False
            return False
    
    def get_faces(self, frame):
        minisize = ( frame.shape[1] / DOWNSCALE, frame.shape[0] / DOWNSCALE )
        miniframe = cv2.resize( frame, minisize )
        faces = self.classifier.detectMultiScale( miniframe )
        return faces

    def extract_and_resize(self, frame, x, y, w, h ):
        cropped = cv2.getRectSubPix( frame, (w,h), ( x+w/2 , y+h/2 ))
        grayscale = cv2.cvtColor( cropped, cv2.COLOR_BGR2GRAY )
        resized = cv2.resize( grayscale, ( IMAGE_SIZE, IMAGE_SIZE ))
        return resized

    def add_to_collection(self, identity, images):
        os.makedirs("{0}/{1}".format( self.images_path, identity))
        idx = 1
        for img in images:
            cv2.imwrite("{0}/{1}/{2}.jpg".format( self.images_path, identity, idx), img)
            idx += 1

    def capture(self):
        cv2.namedWindow("preview")
        capture_num = 1
        captures = []
        i = 0
        resized = None
        rval = True
        while rval:
            time.sleep( self.interval )
            rval, frame = self.camera.read()
            faces = self.get_faces( frame )
            for f in faces:
                x,y,w,h = [ v*DOWNSCALE for v in f ]
                cv2.rectangle( frame, (x,y), (x+w,y+h), (255,0,0))
                resized = self.extract_and_resize( frame, x, y, w, h )
            cv2.putText(frame, "Appuyez sur [ESC] pour quitter.", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))
            cv2.putText(frame, "Appuyez sur [C] pour capturer une photo.", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))
            cv2.imshow("preview", frame)

            key = cv2.waitKey(20)
            if key in [ 27, ord('Q'), ord('q') ]:
                break

            if key in [ ord('C'), ord('c') ]:
                if resized.any() :
                    cv2.imshow( "capture", resized )
                    captures.append( resized )
                    if capture_num >= NUMBER_OF_CAPTURE:
                        return captures
                    capture_num += 1

if __name__ == "__main__":
    C = Collection(images_path = "./images/", interval= 0.1  )
    C.prepare_camera(0)
    if C.is_camera_available():
        captures = C.capture()
        C.add_to_collection( "toto", captures )
