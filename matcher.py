from featureExtractor import FeatureExtractor
import numpy as np
import cv2
import time
class Matcher(object):
    def __init__(self):
        self.extractor = FeatureExtractor()

    def findDifference(self, image1, image2):
        vector1 = self.extractor.extract(image1)
        vector2 = self.extractor.extract(image2)
        return np.linalg.norm(vector1-vector2)

if __name__ == "__main__":
    im1 = cv2.imread("images/car.jpg")
    im2 = cv2.GaussianBlur(im1, (5,5) , 0) 
    im3 = cv2.imread("images/car2.jpeg")
    matcher = Matcher()
    t1 = time.time()
    for _ in range(10):
        print(matcher.findDifference(im1,im2))
        print(matcher.findDifference(im1,im3))
    t2 = time.time() - t1
    print(f"Average Duration is {t2/10}")