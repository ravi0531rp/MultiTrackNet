import cv2
import numpy as np
from multiprocessing import Process, Queue

class Rtsp(object):
    def __init__(self, src = None, camId = None, shouldLoop = False, sampleRate = 5):
        self.src = src
        self.camId = camId
        self.shouldLoop = shouldLoop
        self.sampleRate = sampleRate
        self.endStream = False

    def start(self):
        Process(target=self.streamRtsp).start()

    def stop(self):
        self.endStream = True
    def streamRtsp(self):
        cap = cv2.VideoCapture(self.src)
        if self.shouldLoop:
            while True:
                for _ in range(self.sampleRate):
                    ret, frame = cap.read()
                if ret:
                    
    



