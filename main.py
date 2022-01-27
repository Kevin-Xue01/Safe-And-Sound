import cv2
import numpy as np
from database import Database
from operator import itemgetter
import copy
import serial
import time

with_video = True
with_arduino = False
mapping = {"-2": "out of bounds", "-1": "too low", "1": "too high", "0": "normal"}


def main():
    start_recording_video()


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


def warp_perspective(img: np.ndarray, config_data):
    rows, cols, _ = img.shape
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
    ser = None
    if with_arduino:

        ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
        ser.flush()
    while True:
        tic = time.perf_counter()
        (
            ret,
            frame,
        ) = cap.read()

        frame = cv2.flip(frame, 1)
        s = frame.shape
        frame = frame[:, s[1] // 2 - 160 : s[1] // 2 + 160, :]

        top, bottom = process_green_rectangle(copy.deepcopy(frame), database_controller)
        ball = process_ball(copy.deepcopy(frame), database_controller)
        bnum, bmsg = process_state(top, bottom, ball)
        if with_arduino:

            time.sleep(1)
            ser.write(bytes(f"{bnum}\n", "utf-8"))
            print(f"{bmsg}")
        else:
            # print(f"{bnum}\n")

            pass
        cv2.imshow("camera", frame)
        toc = time.perf_counter()
        print(f"{toc - tic:0.4f},")
        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break

    cap.release()
    cv2.destroyAllWindows()


def apply_rectangle_mask(img, config_data):
    lower_bound, upper_bound = itemgetter("lower_bound", "upper_bound")(config_data)
    lower_bound_np_arr = np.array(lower_bound)
    upper_bound_np_arr = np.array(upper_bound)
    mask = cv2.inRange(img, lower_bound_np_arr, upper_bound_np_arr)
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


if __name__ == "__main__":
    main()
