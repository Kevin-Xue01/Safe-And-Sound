import cv2
import numpy as np
from database import Database
from operator import itemgetter
import copy
from util import Util

with_video = False


def main():
    database_controller = Database()
    util_controller = Util(database_controller=database_controller)
    if with_video:
        start_recording_video()
    else:
        img = util_controller.get_warped_color_image()

        top, bottom = process_green_rectangle(copy.deepcopy(img), database_controller)
        ball = process_ball(copy.deepcopy(img), database_controller
        print(f'Gas flow rate is {process_state(top, bottom, ball)}')
    return


def process_green_rectangle(img, database_controller: Database):

    mask = apply_rectangle_mask(img, database_controller.get_rectangle_mask_data())
    #Util.freeze_current_image(mask, "Mask")
    rectangle_contours = find_green_rectangle_contours(mask)
    top, bottom = find_green_rectangle_polygon(
        rectangle_contours,
        img,
        database_controller.get_rectangle_polygon_contour_data(),
    )
    #Util.freeze_current_image(img, "Find Green Rectangle Polygon")

    return top, bottom


def process_ball(img, database_controller: Database):
    mask = apply_ball_mask(img, database_controller.get_ball_mask_data())
    #Util.freeze_current_image(mask, "Ball Mask")
    ball_contours = find_ball_contours(mask)
    center = find_ball_polygon(ball_contours, img, database_controller.get_ball_polygon_data())

    return center


def process_state(top, bottom, ball):
    if ball ==-1:
        return 'cannot find ball'
    if ball > bottom:
        return 'too low'
    elif ball < top:
        return 'too high'
    else:
        return 'normal'


def warp_perspective(img: np.ndarray, config_data):
    rows, cols, _ = img.shape  # Color
    # rows, cols = img.shape  # Gray scale
    for i in config_data:
        config_data[i] = int(config_data[i])
    (bLC, bLR, bRC, bRR, tLC, tLR, tRC, tRR,) = itemgetter(
        "bLC",
        "bLR",
        "bRC",
        "bRR",
        "tLC",
        "tLR",
        "tRC",
        "tRR",
    )(config_data)

    pts1 = np.float32(
        [
            [int(cols * (tLC / 100)), int(rows * (tLR / 100))],
            [int(cols * (tRC / 100)), int(rows * (tRR / 100))],
            [int(cols * (bLC / 100)), int(rows * (bLR / 100))],
            [int(cols * (bRC / 100)), int(rows * (bRR / 100))],
        ]
    )
    pts2 = np.float32([[cols * 0.1, rows], [cols, rows], [0, 0], [cols, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    distorted = cv2.warpPerspective(img, matrix, (cols, rows))

    return distorted


def start_recording_video():
    cap = cv2.VideoCapture(0)
    database_controller = Database()
    while True:
        (
            ret,
            frame,
        ) = cap.read()
        
        frame = cv2.flip(frame, 1)
        frame = frame[:,s[1]//2-160:s[1]//2+160,:]
        
        top, bottom = process_green_rectangle(copy.deepcopy(frame), database_controller)
        ball = process_ball(copy.deepcopy(frame), database_controller)
        print(f'Gas flow rate is {process_state(top, bottom, ball)}')
        
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break

    cap.release()
    cv2.destroyAllWindows()


def apply_rectangle_mask(img, config_data):
    lower_bound, upper_bound = itemgetter("lower_bound", "upper_bound")(config_data)
    lower_bound_np_arr = np.array(lower_bound)
    upper_bound_np_arr = np.array(upper_bound)
    mask = cv2.inRange(img, lower_bound_np_arr, upper_bound_np_arr)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    return mask


def find_green_rectangle_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_green_rectangle_polygon(contours, img: np.ndarray, config_data):
    rows, _, _ = img.shape

    area_threshold, lower_length_threshold, upper_length_threshold = itemgetter(
        "area_threshold", "lower_length_threshold", "upper_length_threshold"
    )(config_data)
    bottom = 0
    top = rows
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if (
            area > area_threshold
            and len(approx) >= lower_length_threshold
            and len(approx) <= upper_length_threshold
        ):
            cv2.drawContours(img, [approx], 0, (0, 0, 0), 3)  # copy the image
            approx = np.squeeze(approx)
            curr_bottom = np.amax(approx, axis=0)[1]
            curr_top = np.amin(approx, axis=0)[1]

            if curr_bottom > bottom:
                bottom = curr_bottom
            if curr_top < top:
                top = curr_top
    # cv2.line(img, (0, bottom), (cols, bottom), (0, 0, 0), thickness=5)
    # cv2.line(img, (0, top), (cols, top), (0, 0, 0), thickness=5)
    return top, bottom


def apply_ball_mask(img, config_data):

    (
        lower_bound,
        upper_bound,
        blur_kernel_size,
        sigma_x_and_sigma_y,
        erode_and_dilate_kernel_size,
    ) = itemgetter(
        "lower_bound",
        "upper_bound",
        "blur_kernel_size",
        "sigma_x_and_sigma_y",
        "erode_and_dilate_kernel_size",
    )(
        config_data
    )
    blurred = cv2.GaussianBlur(
        img, (blur_kernel_size, blur_kernel_size), sigma_x_and_sigma_y
    )
    lower_bound_np_arr = np.array(lower_bound)
    upper_bound_np_arr = np.array(upper_bound)
    mask = cv2.inRange(blurred, lower_bound_np_arr, upper_bound_np_arr)
    kernel = np.ones(
        (erode_and_dilate_kernel_size, erode_and_dilate_kernel_size), np.uint8
    )
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask


def find_ball_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_ball_polygon(contours, img: np.ndarray, config_data):
    rows, cols, _ = img.shape

    center = None
    (
        area_threshold
    ) = itemgetter(
        "area_threshold"
    )(
        config_data
    )
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        # ((x, y), radius) = cv2.minEnclosingCircle(c)
        # print(x, y, radius)
        if area > area_threshold:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            return -1
        # only proceed if the radius meets a minimum size
        # if radius > 10:
        #     # draw the circle and centroid on the frame,
        #     # then update the list of tracked points
        #     cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        #     cv2.circle(img, center, 5, (0, 0, 255), -1)
    # cv2.line(img, (0, center[1]), (cols, center[1]), (0, 0, 0), thickness=5)
    #cv2.imshow("Ball", img)
    #cv2.waitKey(0)
    
    return center[1] if center else -1


if __name__ == "__main__":
    main()
