from cv2.typing import MatLike
import cv2
import numpy as np

def remove_black_hairline(image: MatLike) -> MatLike:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lowerbound: MatLike = np.array([0, 0, 0])
    upperbound: MatLike = np.array([180, 50, 50])
    black_mask = cv2.inRange(hsv, lowerbound, upperbound)
    result = cv2.inpaint(image, black_mask, 3, cv2.INPAINT_TELEA)
    return result
