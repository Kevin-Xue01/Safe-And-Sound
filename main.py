import cv2 
import numpy as np
import copy

def main():
    start()

def nothing(x):
    pass

def start():
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame, = cap.read()
    #     cv2.imshow('camera', frame)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar("Top Left Column", "Trackbars", 25, 100, nothing)
    cv2.createTrackbar("Top Left Row", "Trackbars", 95, 100, nothing)
    cv2.createTrackbar("Top Right Column", "Trackbars", 90, 100, nothing)
    cv2.createTrackbar("Top Right Row", "Trackbars", 95, 100, nothing)
    cv2.createTrackbar("Bottom Left Column", "Trackbars", 10, 100, nothing)
    cv2.createTrackbar("Bottom Left Row", "Trackbars", 0, 100, nothing)
    cv2.createTrackbar("Bottom Right Column", "Trackbars", 100, 100, nothing)
    cv2.createTrackbar("Bottom Right Row", "Trackbars", 0, 100, nothing)


    img = cv2.imread("assets/main_view.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.Canny(img, 99, 99)
    _, thrash = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    rows, cols = img.shape  
    
    while True:
        newImg = copy.deepcopy(img)
        
        tLC = int(cols * cv2.getTrackbarPos("Top Left Column", "Trackbars") / 100)  
        tLR = int(rows * cv2.getTrackbarPos("Top Left Row", "Trackbars") / 100 )
        tRC = int(cols * cv2.getTrackbarPos("Top Right Column", "Trackbars") / 100) 
        tRR = int(rows * cv2.getTrackbarPos("Top Right Row", "Trackbars") / 100 )
        bLC = int(cols * cv2.getTrackbarPos("Bottom Left Column", "Trackbars") / 100 ) 
        bLR = int(rows * cv2.getTrackbarPos("Bottom Left Row", "Trackbars") / 100 )
        bRC = int(cols * cv2.getTrackbarPos("Bottom Right Column", "Trackbars") / 100) 
        bRR = int(rows * cv2.getTrackbarPos("Bottom Right Row", "Trackbars") / 100 )
        
        exit = cv2.waitKey(1) & 0xFF
        if exit == 27:
            break
        pts1 = np.float32(
            [[tLC, tLR],
             [tRC, tRR],
             [bLC, bLR],
             [bRC, bRR]]
        )
        pts2 = np.float32(
            [[cols*0.1, rows],
             [cols,     rows],
             [0,        0],
             [cols,     0]]
        )    
        
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        distorted = cv2.warpPerspective(newImg, matrix, (cols, rows))
        cv2.imshow('Distorted', distorted)
        cv2.imshow("Original", newImg)
        img = newImg
   
    return

if __name__ == '__main__':
    main()