import cv2 
import numpy as np
import copy
import json

def main():
    img = cv2.imread("assets/main_view.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = None
    with open('config.json', 'r') as file:
        data = json.load(file, parse_int=None)
    img = warp_perspective(img, data)
    # mask = apply_green_mask(img)
    # contours = find_green_rectangle_contours(mask)
    # approximate_polygons_and_draw_contours(contours, img)
    cv2.imshow("Original", img)
    while True:
        if cv2.waitKey(1) == ord('q'): # press q to terminate program
            break    
    pass


def warp_perspective(img, data):
    
    rows, cols = img.shape  
    for i in data:
        data[i] = int(data[i])
    tLC, tLR, tRC, tRR, bLC, bLR, bRC, bRR = data.values()

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
    distorted = cv2.warpPerspective(img, matrix, (cols, rows))
       
    return distorted

def start_recording_video ():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame, = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(frame, 45, 65, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if area > 800:
                cv2.drawContours(frame, [approx], 0, (0,0,0), 5)
                if len(approx) >= 4 and len(approx) <= 10:
                    cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))

        # lower_bound_for_green = np.array([118])
        # upper_bound_for_green = np.array([138])
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # mask = cv2.inRange(frame, lower_bound_for_green, upper_bound_for_green)   
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.erode(mask, kernel)
        # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        #     x = approx.ravel()[0]
        #     y = approx.ravel()[1]
        #     if area > 400:
        #         cv2.drawContours(frame, [approx], 0, (0,0,0), 5)
        #         if len(approx) >= 4 and len(approx) <= 10:
        #             cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
        
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) == ord('q'): # press q to terminate program
            break
        
    cap.release()
    cv2.destroyAllWindows()
    pass

def apply_green_mask(img):
    #
    lower_bound_for_red = np.array([45]) # optimal value is [55, 55, 55]
    upper_bound_for_red = np.array([65])
    mask = cv2.inRange(img, lower_bound_for_red, upper_bound_for_red)   
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    return mask

def find_green_rectangle_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def approximate_polygons_and_draw_contours(contours, img):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if area > 800:
            cv2.drawContours(img, [approx], 0, (0,0,0), 5)
            if len(approx) >= 4 and len(approx) <= 10:
                cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
    pass
def getting_pixel_values():
    img = cv2.imread("assets/main_view.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original", img)
    img = cv2.rectangle(img, (40, 350), (120, 550), (128, 128, 128), 5)
    color = img[250, 30] # 55 the lower the number, the darker
    print(color)
    color = img[350, 40] # 128
    print(color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()