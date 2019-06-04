import numpy as np
import cv2 as cv

with open('/Users/liwenbo/Desktop/人工智能/day01/data/ocr.dat','r') as fp:
    for line in fp.readlines():
        items = line.split('\t')
        char,image = items[1],items[6:-1]
        cv.resize(
            np.array(image,dtype=np.uint8).reshape(16,8)*255,
            None,fx=25,fy=25)
        cv.imshow(char,image)
        if cv.waitKey(100) == 27:
            break