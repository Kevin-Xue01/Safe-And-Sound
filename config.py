import cv2 
import numpy as np
import copy
from database import Database
from operator import itemgetter
from main import *

database_controller = Database()
with_frame_flip = True
with_frame_crop = True


def main():
    cap = cv2.VideoCapture(0)

    change_rectangle_mask_color_threshold(cap)
    change_rectangle_polygon_contour_params(cap)
    change_ball_mask_params(cap)

    return

def change_rectangle_mask_color_threshold(cap):

    lower_bound, upper_bound = itemgetter("lower_bound", "upper_bound")(
        database_controller.get_rectangle_mask_data()
    )

    def nothing(x):
        pass

    cv2.namedWindow("Green Mask Trackbar")
    cv2.resizeWindow("Green Mask Trackbar", 800, 600)  # width, height
    cv2.createTrackbar(
        "Lower Blue Threshold", "Green Mask Trackbar", lower_bound[0], 255, nothing
    )
    cv2.createTrackbar(
        "Lower Green Threshold", "Green Mask Trackbar", lower_bound[1], 255, nothing
    )
    cv2.createTrackbar(
        "Lower Red Threshold", "Green Mask Trackbar", lower_bound[2], 255, nothing
    )
    cv2.createTrackbar(
        "Upper Blue Threshold", "Green Mask Trackbar", upper_bound[0], 255, nothing
    )
    cv2.createTrackbar(
        "Upper Green Threshold", "Green Mask Trackbar", upper_bound[1], 255, nothing
    )
    cv2.createTrackbar(
        "Upper Red Threshold", "Green Mask Trackbar", upper_bound[2], 255, nothing
    )
    mask = None
    while True:
        ret, frame = cap.read()
        rows, cols, _ = frame.shape
        if with_frame_flip:
            frame = cv2.flip(frame, 1)

        if with_frame_crop:
            frame = frame[:, cols // 2 - 160 : cols // 2 + 160, :]

        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break
        newMask = copy.deepcopy(mask)
        lower_blue_threshold = cv2.getTrackbarPos(
            "Lower Blue Threshold", "Green Mask Trackbar"
        )
        lower_green_threshold = cv2.getTrackbarPos(
            "Lower Green Threshold", "Green Mask Trackbar"
        )
        lower_red_threshold = cv2.getTrackbarPos(
            "Lower Red Threshold", "Green Mask Trackbar"
        )
        upper_blue_threshold = cv2.getTrackbarPos(
            "Upper Blue Threshold", "Green Mask Trackbar"
        )
        upper_green_threshold = cv2.getTrackbarPos(
            "Upper Green Threshold", "Green Mask Trackbar"
        )
        upper_red_threshold = cv2.getTrackbarPos(
            "Upper Red Threshold", "Green Mask Trackbar"
        )
        lower_bound_for_green = np.array(
            [lower_blue_threshold, lower_green_threshold, lower_red_threshold]
        )
        upper_bound_for_green = np.array(
            [upper_blue_threshold, upper_green_threshold, upper_red_threshold]
        )
        newMask = cv2.inRange(frame, lower_bound_for_green, upper_bound_for_green)
        kernel = np.ones((1, 1), np.uint8)
        newMask = cv2.erode(newMask, kernel)
        cv2.imshow("Green Mask", newMask)
        cv2.imshow("Frame", frame)
        mask = newMask

    database_controller.update_rectangle_mask_data(
        {
            "lower_bound": [
                lower_blue_threshold,
                lower_green_threshold,
                lower_red_threshold,
            ],
            "upper_bound": [
                upper_blue_threshold,
                upper_green_threshold,
                upper_red_threshold,
            ],
        }
    )
    cv2.destroyAllWindows()


def change_rectangle_polygon_contour_params(cap):

    area_threshold, lower_length_threshold, upper_length_threshold = itemgetter(
        "area_threshold", "lower_length_threshold", "upper_length_threshold"
    )(database_controller.get_rectangle_polygon_contour_data())

    def nothing(x):
        pass

    cv2.namedWindow("Rectangle Polygon Contour Trackbar")
    cv2.resizeWindow("Rectangle Polygon Contour Trackbar", 800, 600)  # width, height
    cv2.createTrackbar(
        "Area Threshold",
        "Rectangle Polygon Contour Trackbar",
        area_threshold,
        5000,
        nothing,
    )
    cv2.createTrackbar(
        "Lower Length Threshold",
        "Rectangle Polygon Contour Trackbar",
        lower_length_threshold,
        255,
        nothing,
    )
    cv2.createTrackbar(
        "Upper Length Threshold",
        "Rectangle Polygon Contour Trackbar",
        upper_length_threshold,
        255,
        nothing,
    )

    while True:
        ret, frame = cap.read()
        rows, cols, _ = frame.shape

        if with_frame_flip:
            frame = cv2.flip(frame, 1)

        if with_frame_crop:
            frame = frame[:, cols // 2 - 160 : cols // 2 + 160, :]

        contours = find_green_rectangle_contours(
            apply_rectangle_mask(frame, database_controller.get_rectangle_mask_data())
        )
        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break
        area_threshold = cv2.getTrackbarPos(
            "Area Threshold", "Rectangle Polygon Contour Trackbar"
        )
        lower_length_threshold = cv2.getTrackbarPos(
            "Lower Length Threshold", "Rectangle Polygon Contour Trackbar"
        )
        upper_length_threshold = cv2.getTrackbarPos(
            "Upper Length Threshold", "Rectangle Polygon Contour Trackbar"
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)

            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if (
                area > area_threshold
                and len(approx) >= lower_length_threshold
                and len(approx) <= upper_length_threshold
            ):
                cv2.drawContours(frame, [approx], 0, (0, 0, 0), 3)
        cv2.imshow("Original", frame)

    database_controller.update_rectangle_polygon_contour_data(
        {
            "area_threshold": area_threshold,
            "lower_length_threshold": lower_length_threshold,
            "upper_length_threshold": upper_length_threshold,
        }
    )

    cv2.destroyAllWindows()


def change_ball_mask_params(cap):

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
        database_controller.get_ball_mask_data()
    )
    area_threshold = itemgetter("area_threshold")(
        database_controller.get_ball_polygon_data()
    )

    def nothing(x):
        pass

    cv2.namedWindow("Ball Mask Trackbar")
    cv2.resizeWindow("Ball Mask Trackbar", 800, 600)  # width, height
    cv2.createTrackbar(
        "Lower Blue Threshold", "Ball Mask Trackbar", lower_bound[0], 255, nothing
    )
    cv2.createTrackbar(
        "Lower Green Threshold", "Ball Mask Trackbar", lower_bound[1], 255, nothing
    )
    cv2.createTrackbar(
        "Lower Red Threshold", "Ball Mask Trackbar", lower_bound[2], 255, nothing
    )
    cv2.createTrackbar(
        "Upper Blue Threshold", "Ball Mask Trackbar", upper_bound[0], 255, nothing
    )
    cv2.createTrackbar(
        "Upper Green Threshold", "Ball Mask Trackbar", upper_bound[1], 255, nothing
    )
    cv2.createTrackbar(
        "Upper Red Threshold", "Ball Mask Trackbar", upper_bound[2], 255, nothing
    )
    cv2.createTrackbar(
        "Area Threshold", "Ball Mask Trackbar", area_threshold, 1000, nothing
    )

    mask = None
    while True:
        ret, frame = cap.read()
        rows, cols, _ = frame.shape

        if with_frame_flip:
            frame = cv2.flip(frame, 1)

        if with_frame_crop:
            frame = frame[:, cols // 2 - 160 : cols // 2 + 160, :]

        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break
        newMask = copy.deepcopy(mask)
        lower_blue_threshold = cv2.getTrackbarPos(
            "Lower Blue Threshold", "Ball Mask Trackbar"
        )
        lower_green_threshold = cv2.getTrackbarPos(
            "Lower Green Threshold", "Ball Mask Trackbar"
        )
        lower_red_threshold = cv2.getTrackbarPos(
            "Lower Red Threshold", "Ball Mask Trackbar"
        )
        upper_blue_threshold = cv2.getTrackbarPos(
            "Upper Blue Threshold", "Ball Mask Trackbar"
        )
        upper_green_threshold = cv2.getTrackbarPos(
            "Upper Green Threshold", "Ball Mask Trackbar"
        )
        upper_red_threshold = cv2.getTrackbarPos(
            "Upper Red Threshold", "Ball Mask Trackbar"
        )
        area_threshold = cv2.getTrackbarPos("Area Threshold", "Ball Mask Trackbar")

        blurred = cv2.GaussianBlur(
            frame, (blur_kernel_size, blur_kernel_size), sigma_x_and_sigma_y
        )
        lower_bound_for_green = np.array(
            [lower_blue_threshold, lower_green_threshold, lower_red_threshold]
        )
        upper_bound_for_green = np.array(
            [upper_blue_threshold, upper_green_threshold, upper_red_threshold]
        )
        newMask = cv2.inRange(blurred, lower_bound_for_green, upper_bound_for_green)
        kernel = np.ones(
            (erode_and_dilate_kernel_size, erode_and_dilate_kernel_size), np.uint8
        )
        newMask = cv2.erode(newMask, kernel)
        newMask = cv2.dilate(newMask, kernel)

        contours = find_ball_contours(newMask)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > area_threshold:
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                return -1
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        cv2.imshow("Ball Polygon", frame)
        cv2.imshow("Ball Mask", newMask)
        mask = newMask

    database_controller.update_ball_mask_data(
        {
            "lower_bound": [
                lower_blue_threshold,
                lower_green_threshold,
                lower_red_threshold,
            ],
            "upper_bound": [
                upper_blue_threshold,
                upper_green_threshold,
                upper_red_threshold,
            ],
            "blur_kernel_size": blur_kernel_size,
            "sigma_x_and_sigma_y": sigma_x_and_sigma_y,
            "erode_and_dilate_kernel_size": erode_and_dilate_kernel_size,
        }
    )
    database_controller.update_ball_polygon_data({"area_threshold": area_threshold})

    cv2.destroyAllWindows()

def find_ball_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _ = 1
    return contours


if __name__ == "__main__":
    main()
