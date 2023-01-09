import numpy as np
import cv2 as cv

def colorful (path, K):
    image = cv.imread(path)
    data = image.reshape((-1, 3))
    data = np.float32(data)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 2)
    ret,label,center=cv.kmeans(data, K, None, criteria, K, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)

    color = np.uint8([[255, 0, 0],
                  [0, 0, 255],
                  [0, 255, 0],
                  [128, 128, 128],
                  [255,255,0],
                  [0,255,255],
                  [255,0,255],
                  [255,255,255]])
    res = color[label.flatten()]

    result = res.reshape((image.shape))
    cv.imshow('DD', result)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__" :
    path = r"IMG_7025/.jpg"
    K = 3
    colorful(path, K)