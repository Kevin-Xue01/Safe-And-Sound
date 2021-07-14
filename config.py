import cv2
import numpy as np
import copy
import json
from database import Database
from operator import itemgetter
from util import Util
from main import *

database_controller = Database()


def main():

    img = Util.get_color_image()
    # get_pixel_values(img)
    # change_perspective_warp_params(img)
    # img = Util.warp_perspective(img)
    # ball_edge_detection_config(img)
    # change_rectangle_mask_color_threshold(img)
    # change_rectangle_polygon_contour_params(img)
    change_ball_mask_params(img)

    return


def change_perspective_warp_params(img: np.ndarray):
    def nothing(x):
        pass

    (bLC, bLR, bRC, bRR, tLC, tLR, tRC, tRR,) = itemgetter(
        "bLC",
        "bLR",
        "bRC",
        "bRR",
        "tLC",
        "tLR",
        "tRC",
        "tRR",
    )(database_controller.get_warp_perspective_data())

    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 1000, 400)  # width, height
    cv2.createTrackbar("Top Left Column", "Trackbars", tLC, 100, nothing)
    cv2.createTrackbar("Top Left Row", "Trackbars", tLR, 100, nothing)
    cv2.createTrackbar("Top Right Column", "Trackbars", tRC, 100, nothing)
    cv2.createTrackbar("Top Right Row", "Trackbars", tRR, 100, nothing)
    cv2.createTrackbar("Bottom Left Column", "Trackbars", bLC, 100, nothing)
    cv2.createTrackbar("Bottom Left Row", "Trackbars", bLR, 100, nothing)
    cv2.createTrackbar("Bottom Right Column", "Trackbars", bRC, 100, nothing)
    cv2.createTrackbar("Bottom Right Row", "Trackbars", bRR, 100, nothing)

    rows, cols, _ = img.shape

    while True:
        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break
        newImg = copy.deepcopy(img)

        tLC = cv2.getTrackbarPos("Top Left Column", "Trackbars")
        tLR = cv2.getTrackbarPos("Top Left Row", "Trackbars")
        tRC = cv2.getTrackbarPos("Top Right Column", "Trackbars")
        tRR = cv2.getTrackbarPos("Top Right Row", "Trackbars")
        bLC = cv2.getTrackbarPos("Bottom Left Column", "Trackbars")
        bLR = cv2.getTrackbarPos("Bottom Left Row", "Trackbars")
        bRC = cv2.getTrackbarPos("Bottom Right Column", "Trackbars")
        bRR = cv2.getTrackbarPos("Bottom Right Row", "Trackbars")

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
        distorted = cv2.warpPerspective(newImg, matrix, (cols, rows))
        cv2.imshow("Distorted", distorted)
        cv2.imshow("Original", newImg)
        img = newImg

    database_controller.update_warp_perspective_data(
        {
            "bLC": bLC,
            "bLR": bLR,
            "bRC": bRC,
            "bRR": bRR,
            "tLC": tLC,
            "tLR": tLR,
            "tRC": tRC,
            "tRR": tRR,
        }
    )
    cv2.destroyAllWindows()


def get_pixel_values(img: np.ndarray):
    row, col, _ = img.shape

    def nothing(x):
        pass

    cv2.namedWindow("Pixel Value Trackbar")
    cv2.resizeWindow("Pixel Value Trackbar", 600, 200)  # width, height
    cv2.createTrackbar("Row Number", "Pixel Value Trackbar", 350, row, nothing)
    cv2.createTrackbar("Column Number", "Pixel Value Trackbar", 30, col, nothing)
    row_number = 0
    col_number = 0
    while True:
        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break
        newImg = copy.deepcopy(img)

        row_number = cv2.getTrackbarPos("Row Number", "Pixel Value Trackbar")
        col_number = cv2.getTrackbarPos("Column Number", "Pixel Value Trackbar")
        cv2.circle(newImg, (col_number, row_number), 5, (0, 0, 0), 1)

        cv2.imshow("Getting Pixel Value", newImg)
    cv2.destroyAllWindows()
    print(
        f"Pixel value at Col:{col_number}, Row: {row_number}, is {img[row_number][col_number]}"
    )
    return


# def ball_edge_detection_config(img: np.ndarray):

#     lower, upper = itemgetter("lower", "upper")(
#         database_controller.get_ball_edge_detection_data()
#     )

#     def nothing(x):
#         pass

#     cv2.namedWindow("Ball Edge Detection Trackbar")
#     cv2.resizeWindow("Ball Edge Detection Trackbar", 800, 100)  # width, height
#     cv2.createTrackbar(
#         "Lower Threshold", "Ball Edge Detection Trackbar", lower, 1000, nothing
#     )
#     cv2.createTrackbar(
#         "Upper Threshold", "Ball Edge Detection Trackbar", upper, 1000, nothing
#     )
#     canny = None
#     while True:
#         if cv2.waitKey(1) == ord("q"):  # press q to terminate program
#             break
#         newImg = copy.deepcopy(img)

#         lower = cv2.getTrackbarPos("Lower Threshold", "Ball Edge Detection Trackbar")
#         upper = cv2.getTrackbarPos("Upper Threshold", "Ball Edge Detection Trackbar")

#         canny = cv2.Canny(newImg, lower, upper)

#         img = newImg
#         cv2.imshow("Canny", canny)
#     database_controller.update_ball_edge_detection_data(
#         {"lower": lower, "upper": upper}
#     )
#     cv2.destroyAllWindows()


def change_rectangle_mask_color_threshold(img: np.ndarray):

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
        newMask = cv2.inRange(img, lower_bound_for_green, upper_bound_for_green)
        kernel = np.ones((5, 5), np.uint8)
        newMask = cv2.erode(newMask, kernel)
        cv2.imshow("Green Mask", newMask)
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


def change_rectangle_polygon_contour_params(img: np.ndarray):
    contours = find_green_rectangle_contours(
        apply_rectangle_mask(img, database_controller.get_rectangle_mask_data())
    )

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
        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break
        newImg = copy.deepcopy(img)
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
                print(area)
                cv2.drawContours(newImg, [approx], 0, (0, 0, 0), 3)
        cv2.imshow("Original", newImg)

    database_controller.update_rectangle_polygon_contour_data(
        {
            "area_threshold": area_threshold,
            "lower_length_threshold": lower_length_threshold,
            "upper_length_threshold": upper_length_threshold,
        }
    )

    cv2.destroyAllWindows()


def change_ball_mask_params(img: np.ndarray):

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

    mask = None
    while True:
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

        blurred = cv2.GaussianBlur(
            img, (blur_kernel_size, blur_kernel_size), sigma_x_and_sigma_y
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
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
