import cv2 
import numpy as np
import copy
import json

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def main():
    img = cv2.imread("assets/main_view.jpg")
    get_pixel_values(img)
    # warp_perspective_config(img)
    # while True:
    #     if cv2.waitKey(1) == ord('q'): # press q to terminate program
    #         break    
    # cv2.destroyAllWindows()
    return


def warp_perspective_config(img):
    def nothing(x):
        pass
    
    data = None
    with open('config.json', 'r') as file:
        data = json.load(file, parse_int=None)
        
    tLC, tLR, tRC, tRR, bLC, bLR, bRC, bRR = data.values()

    cv2.namedWindow('Trackbars')
    cv2.createTrackbar("Top Left Column", "Trackbars", tLC, 100, nothing)
    cv2.createTrackbar("Top Left Row", "Trackbars", tLR, 100, nothing)
    cv2.createTrackbar("Top Right Column", "Trackbars", tRC, 100, nothing)
    cv2.createTrackbar("Top Right Row", "Trackbars", tRR, 100, nothing)
    cv2.createTrackbar("Bottom Left Column", "Trackbars", bLC, 100, nothing)
    cv2.createTrackbar("Bottom Left Row", "Trackbars", bLR, 100, nothing)
    cv2.createTrackbar("Bottom Right Column", "Trackbars", bRC, 100, nothing)
    cv2.createTrackbar("Bottom Right Row", "Trackbars", bRR, 100, nothing)

    rows, cols, _ = img.shape  
    # rows, cols = img.shape  
    
    while True:
        newImg = copy.deepcopy(img)
        
        tLC = cv2.getTrackbarPos("Top Left Column", "Trackbars") 
        tLR = cv2.getTrackbarPos("Top Left Row", "Trackbars") 
        tRC = cv2.getTrackbarPos("Top Right Column", "Trackbars") 
        tRR = cv2.getTrackbarPos("Top Right Row", "Trackbars") 
        bLC = cv2.getTrackbarPos("Bottom Left Column", "Trackbars")  
        bLR = cv2.getTrackbarPos("Bottom Left Row", "Trackbars") 
        bRC = cv2.getTrackbarPos("Bottom Right Column", "Trackbars") 
        bRR = cv2.getTrackbarPos("Bottom Right Row", "Trackbars") 
        
        if cv2.waitKey(1) == ord('q'): # press q to terminate program
            break
        pts1 = np.float32(
            [[int(cols*(tLC/100)), int(rows * (tLR/100))],
             [int(cols*(tRC/100)), int(rows * (tRR/100))],
             [int(cols*(bLC/100)), int(rows * (bLR/100))],
             [int(cols*(bRC/100)), int(rows * (bRR/100))]]
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

    data = {"tLC": tLC, "tLR": tLR, "tRC": tRC, "tRR": tRR, "bLC": bLC, "bLR": bLR, "bRC": bRC, "bRR": bRR}
    with open('config.json', 'w') as file:
        json.dump(data, file)
    return

def get_pixel_values(img):
    row, col, _ = img.shape
    def nothing(x):
        pass
    cv2.namedWindow('Pixel Value Trackbar')
    cv2.resizeWindow("Pixel Value Trackbar", 600, 200) # width, height
    cv2.createTrackbar("Row Number", "Pixel Value Trackbar", 350, row, nothing)
    cv2.createTrackbar("Column Number", "Pixel Value Trackbar", 30, col, nothing)

    while True:
        if cv2.waitKey(1) == ord('q'): # press q to terminate program
            break
        newImg = copy.deepcopy(img)

        row_number = cv2.getTrackbarPos("Row Number", "Pixel Value Trackbar") 
        col_number = cv2.getTrackbarPos("Column Number", "Pixel Value Trackbar") 
        cv2.circle(newImg, (col_number, row_number), 5, (0,0,0), 1)
        
        cv2.imshow('Getting Pixel Value', newImg)
    cv2.destroyAllWindows()
    return

def ball_edge_detection_config(img):
    def nothing(x):
        pass
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar("Lower Threshold", "Trackbars", 315, 1000, nothing)
    cv2.createTrackbar("Upper Threshold", "Trackbars", 0, 1000, nothing)
    canny = None
    while True:
        newImg = copy.deepcopy(img)
    
        lower = cv2.getTrackbarPos("Lower Threshold", "Trackbars") 
        upper = cv2.getTrackbarPos("Upper Threshold", "Trackbars") 
        
        canny = cv2.Canny(newImg, lower, upper)
        if cv2.waitKey(1) == ord('q'): # press q to terminate program
            break
        img = newImg
        cv2.imshow('Canny', canny)
    return canny

if __name__ == '__main__':
    main()