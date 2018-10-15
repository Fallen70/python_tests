import cv2
import numpy
import time

TRAINSET = "/usr/share/opencv/lbpcascades/lbpcascade_frontalface.xml"
DOWNSCALE = 4

class Detection():

    def __init__(self, interval = 0.1 ):
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

    def capture(self):
        cv2.namedWindow("preview")
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
            cv2.putText(frame, "Appuyez sur [ESC] pour quitter.", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))
            cv2.imshow("preview", frame)

            key = cv2.waitKey(20)
            if key in [ 27, ord('Q'), ord('q') ]:
                break

if __name__ == "__main__":
    D = Detection()
    D.prepare_camera(0)
    if D.is_camera_available():
        D.capture()
