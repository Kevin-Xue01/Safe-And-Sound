import cv2
import numpy as np
import copy
import serial
import time
from operator import itemgetter
from database import Database


def process_green_rectangle(img, database_controller: Database):
    mask = apply_rectangle_mask(img, database_controller.get_rectangle_mask_data())
    rectangle_contours = find_green_rectangle_contours(mask)
    top, bottom = find_green_rectangle_polygon(
        rectangle_contours,
        img,
        database_controller.get_rectangle_polygon_contour_data(),
    )
    return top, bottom


def process_ball(img, database_controller: Database):
    mask = apply_ball_mask(img, database_controller.get_ball_mask_data())
    ball_contours = find_ball_contours(mask)
    center = find_ball_polygon(
        ball_contours, img, database_controller.get_ball_polygon_data()
    )
    return center


def process_state(top, bottom, ball):
    if ball == -1:
        return "-2", "not found"
    if ball > bottom:
        return "-1", "too low"
    elif ball < top:
        return "1", "too high"
    else:
        return "0", "normal"


def apply_rectangle_mask(img, config_data):
    lower_bound, upper_bound = itemgetter("lower_bound", "upper_bound")(config_data)
    lower_bound_np_arr = np.array(lower_bound)
    upper_bound_np_arr = np.array(upper_bound)
    mask = cv2.inRange(img, lower_bound_np_arr, upper_bound_np_arr)
    # noise removal
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
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

    center = None
    (area_threshold) = itemgetter("area_threshold")(config_data)
    if len(contours) > 0:

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > area_threshold:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            return -1

    return center[1] if center else -1

def start_recording_video():
    with_arduino = True

    cap = cv2.VideoCapture(0)
    database_controller = Database()
    ser = None
    if with_arduino:
        ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
        ser.flush()
    while True:
        (
            _,
            frame,
        ) = cap.read()

        frame = cv2.flip(frame, 1)
        s = frame.shape # (height, width, depth)
        frame = frame[:, s[1] // 2 - 160 : s[1] // 2 + 160, :]

        top, bottom = process_green_rectangle(frame.copy(), database_controller)
        ball = process_ball(frame.copy(), database_controller)
        output_numeric_value, output_string_value = process_state(top, bottom, ball)
        if with_arduino:
            time.sleep(1)
            ser.write(bytes(f"{output_numeric_value}\n", "utf-8"))
            print(f"{output_string_value}")
    
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_recording_video()
