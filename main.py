import cv2 
import numpy as np
import copy

def main():
    # warp_perspective()
    # start_recording_video()
    getting_pixel_values()

def nothing(x):
    pass

def warp_perspective():
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

def start_recording_video ():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame, = cap.read()
        lower_red = np.array([0, 68, 154])
        upper_red = np.array([180, 255, 243])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)   
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if area > 400:
                cv2.drawContours(frame, [approx], 0, (0,0,0), 5)
                if len(approx) >= 4 and len(approx) <= 10:
                    cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
        
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    pass

def getting_pixel_values():
    img = cv2.imread("assets/main_view.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.Canny(img, 99, 99)
    _, thrash = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    rows,cols = img.shape
    print(rows, cols)
    cv2.imshow("Original", img)
    img = cv2.rectangle(img, (40, 350), (120, 550), (128, 128, 128), 5)
    color = img[250, 30]
    print(color)
    color = img[350, 40]
    print(color)

    while True:
        exit = cv2.waitKey(1) & 0xFF
        if exit == 27:
            break
        
        cv2.imshow("Original", img)
   
    return

if __name__ == '__main__':
    main()