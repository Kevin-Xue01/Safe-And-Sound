import cv2
from cv2 import data
import numpy as np
import copy
import json
from database import Database
from operator import itemgetter


def main():
    database_controller = Database()
    img = cv2.imread("assets/main_view.jpg")
    get_pixel_values(img)
    warp_perspective_config(img, database_controller)
    ball_edge_detection_config(img, database_controller)
    apply_green_mask(img)
    return


def warp_perspective_config(img: np.ndarray, database_controller: Database):
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
        newImg = copy.deepcopy(img)

        tLC = cv2.getTrackbarPos("Top Left Column", "Trackbars")
        tLR = cv2.getTrackbarPos("Top Left Row", "Trackbars")
        tRC = cv2.getTrackbarPos("Top Right Column", "Trackbars")
        tRR = cv2.getTrackbarPos("Top Right Row", "Trackbars")
        bLC = cv2.getTrackbarPos("Bottom Left Column", "Trackbars")
        bLR = cv2.getTrackbarPos("Bottom Left Row", "Trackbars")
        bRC = cv2.getTrackbarPos("Bottom Right Column", "Trackbars")
        bRR = cv2.getTrackbarPos("Bottom Right Row", "Trackbars")

        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break
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

    return


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
    print(f"Pixel value is Col:{col_number}, Row: {row_number}")
    return


def ball_edge_detection_config(img: np.ndarray, database_controller: Database):

    lower, upper = itemgetter("lower", "upper")(
        database_controller.get_ball_edge_detection_data()
    )

    def nothing(x):
        pass

    cv2.namedWindow("Ball Edge Detection Trackbar")
    cv2.resizeWindow("Ball Edge Detection Trackbar", 800, 100)  # width, height
    cv2.createTrackbar(
        "Lower Threshold", "Ball Edge Detection Trackbar", lower, 1000, nothing
    )
    cv2.createTrackbar(
        "Upper Threshold", "Ball Edge Detection Trackbar", upper, 1000, nothing
    )
    canny = None
    while True:
        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break
        newImg = copy.deepcopy(img)

        lower = cv2.getTrackbarPos("Lower Threshold", "Ball Edge Detection Trackbar")
        upper = cv2.getTrackbarPos("Upper Threshold", "Ball Edge Detection Trackbar")

        canny = cv2.Canny(newImg, lower, upper)

        img = newImg
        cv2.imshow("Canny", canny)
    database_controller.update_ball_edge_detection_data(
        {"lower": lower, "upper": upper}
    )

    return canny


def apply_green_mask(img):
    data = None
    with open("config.json", "r") as file:
        data = json.load(file, parse_int=None)

    lower_bound, upper_bound = data["green_mask"].values()

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

    data["green_mask"] = {
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
    with open("config.json", "w") as file:
        json.dump(data, file, sort_keys=True, indent=4)
    return mask


if __name__ == "__main__":
    main()
