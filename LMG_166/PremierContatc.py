import cv2

class PremierContact():

    def __init__(self):
        self.camera = None

    def prepare_camera(self, capture_device):
        self.camera = cv2.VideoCapture( capture_device )

    def is_camera_available(self):
        if self.camera.isOpened():
            rval, frame = self.camera.read()
            return rval
        else:
            rval = False
            return False

    def capture(self):
        cv2.namedWindow("preview")
        rval = True
        while rval:
            rval, frame = self.camera.read()
            cv2.putText(frame, "Appuyez sur [ESC] pour quitter.", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))
            cv2.imshow("preview", frame)

            key = cv2.waitKey(20)
            if key in [ 27, ord('Q'), ord('q') ]:
                break

if __name__ == "__main__":
    PC = PremierContact()
    PC.prepare_camera(0)
    if PC.is_camera_available():
        PC.capture()
