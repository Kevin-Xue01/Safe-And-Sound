import cv2 
import numpy as np

def main():
    start()

def nothing(x):
    pass

def start():
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar("A", "Trackbars", 0, 100, nothing)
    cv2.createTrackbar("B", "Trackbars", 91, 100, nothing)
    cv2.createTrackbar("C", "Trackbars", 100, 100, nothing)
    cv2.createTrackbar("D", "Trackbars", 90, 100, nothing)
    cv2.createTrackbar("E", "Trackbars", 0, 100, nothing)
    cv2.createTrackbar("F", "Trackbars", 0, 100, nothing)

    img = cv2.imread("assets/main_view.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 99, 99)
    rows, cols = img.shape  
    
    while True:
        a = cv2.getTrackbarPos("A", "Trackbars") / 100  
        b = cv2.getTrackbarPos("B", "Trackbars") / 100 
        c = cv2.getTrackbarPos("C", "Trackbars") / 100 
        d = cv2.getTrackbarPos("D", "Trackbars") / 100 
        e = cv2.getTrackbarPos("E", "Trackbars") / 100 
        f = cv2.getTrackbarPos("F", "Trackbars") / 100 
        exit = cv2.waitKey(1) & 0xFF
        if exit == 27:
            break
        pts1 = np.float32(
            [[cols*a, rows*b],
             [cols*c, rows*d],
             [cols*e, 0],
             [cols,     0]]
        )
        pts2 = np.float32(
            [[cols*f, rows],
             [cols,     rows],
             [0,        0],
             [cols,     0]]
        )    
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        distorted = cv2.warpPerspective(img, matrix, (cols, rows))
        cv2.imshow('Distorted', distorted)
        cv2.imshow("Original", img)
   
    return

if __name__ == '__main__':
    main()