import sys
import cv2
import numpy
import time
import os

TRAINSET = "/usr/share/opencv/lbpcascades/lbpcascade_frontalface.xml"
DOWNSCALE = 4
IMAGE_SIZE = 170
NUMBER_OF_CAPTURE = 10

class Recognize():

    def __init__(self, images_path, interval = 0.1 ):
        self.images_path = images_path
        self.identities = []
        self.images = []
        self.images_index = []
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

    def train(self ):
        self.model = cv2.face.createLBPHFaceRecognizer()
        #self.model = cv2.face.createEigenFaceRecognizer()
        self.model.train(numpy.asarray(self.images), numpy.asarray(self.images_index))

    def identify(self, image):
        [p_index, p_confidence] = self.model.predict(image)
        found_identity = self.identities[p_index]
        return found_identity, p_confidence

    def recognize( self, camera=0, callback=None, view=True):
        self.read_images()
        self.train()
        self.prepare_camera(camera)
        if not self.is_camera_available():
            return
        self.capture(mode='identify', callback=callback, view=view)

    def acquire( self, identity, camera=0 ):
        self.prepare_camera(camera)
        if not self.is_camera_available():
            return
        captures = self.capture( mode='acquire')
        self.add_to_collection( identity, captures )

    def read_images( self, sz=None ):
        c = 0
        self.images = []
        self.images_index = []
        for dirname, dirnames, filenames in os.walk( self.images_path ):
            for subdirname in dirnames:
                self.identities.append( subdirname )
                subject_path = os.path.join( dirname, subdirname )
                for filename in os.listdir( subject_path ):
                    try:
                        im = cv2.imread( os.path.join( subject_path, filename ), cv2.IMREAD_GRAYSCALE)
                        if ( sz is not None ):
                            im = cv2.resize( im, sz )
                        self.images.append( numpy.asarray( im, dtype=numpy.uint8))
                        self.images_index.append(c)
                    except IOError, ( errno, strerror ):
                        print " I/O Error({0}): {1}".format(errno, strerror)
                    except:
                        print "Unexpected error:", sys.exc_info()[0]
                        raise
                c=c+1

    def capture(self, mode, callback=None, view=True):
        if view:
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
                if mode == "identify":
                    identity, confidence = self.identify(resized)
                    cv2.putText( frame, "%s (%s)" % (identity, confidence), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))
                    callback(identity, confidence)
            cv2.putText(frame, "Appuyez sur [ESC] pour quitter.", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))
            if mode == "acquire":
                cv2.putText(frame, "Appuyez sur [C] pour capturer une photo.", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))
            if view:
                cv2.imshow("preview", frame)

            key = cv2.waitKey(20)
            if key in [ 27, ord('Q'), ord('q') ]:
                break

            if mode == "acquire" and key in [ ord('C'), ord('c') ]:
                if resized.any() :
                    cv2.imshow( "capture", resized )
                    captures.append( resized )
                    if capture_num >= NUMBER_OF_CAPTURE:
                        return captures
                    capture_num += 1

def display_recognize( identity, confidence ):
    print( "identity = {0} (confidence = {1}".format( identity, confidence ))

if __name__ == "__main__":
    R = Recognize(images_path = "./images/", interval= 0.1  )
    action = sys.argv[1]
    if action == "acquire":
        R.acquire(camera = 0, identity = sys.argv[2] )
    if action == "reco":
        R.recognize( camera = 0, callback = display_recognize, view = True )
