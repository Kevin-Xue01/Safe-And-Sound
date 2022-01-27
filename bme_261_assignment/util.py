# import cv2
# import numpy as np
# from operator import itemgetter
# from database import Database


# class Util:
#     def __init__(
#         self, database_controller=Database(), image_location="assets/main_view.jpg"
#     ):
#         self.database_controller = database_controller
#         self.image_location = image_location

#     def get_color_image(self):
#         img = cv2.imread(self.image_location)
#         return img

#     def get_grayscale_image(self):
#         img = cv2.imread(self.image_location, 0)
#         return img

#     def freeze_current_image_array(images, titles):
#         for index in range(len(images)):
#             cv2.imshow(titles[index], images[index])
#         while True:
#             if cv2.waitKey(1) == ord("q"):  # press q to terminate program
#                 break
#         cv2.destroyAllWindows()

#     def freeze_current_image(image: np.ndarray, title="Untitled"):
#         cv2.imshow(title, image)
#         while True:
#             if cv2.waitKey(1) == ord("q"):  # press q to terminate program
#                 break
#         cv2.destroyAllWindows()

#     def get_warped_color_image(self):
#         img = cv2.imread(self.image_location)
#         return self.warp_perspective(img)

#     def warp_perspective(self, img: np.ndarray):
#         config_data = self.database_controller.get_warp_perspective_data()
#         rows, cols, _ = img.shape  # Color
#         # rows, cols = img.shape  # Gray scale
#         for i in config_data:
#             config_data[i] = int(config_data[i])
#         (bLC, bLR, bRC, bRR, tLC, tLR, tRC, tRR,) = itemgetter(
#             "bLC",
#             "bLR",
#             "bRC",
#             "bRR",
#             "tLC",
#             "tLR",
#             "tRC",
#             "tRR",
#         )(config_data)

#         pts1 = np.float32(
#             [
#                 [int(cols * (tLC / 100)), int(rows * (tLR / 100))],
#                 [int(cols * (tRC / 100)), int(rows * (tRR / 100))],
#                 [int(cols * (bLC / 100)), int(rows * (bLR / 100))],
#                 [int(cols * (bRC / 100)), int(rows * (bRR / 100))],
#             ]
#         )
#         pts2 = np.float32([[cols * 0.1, rows], [cols, rows], [0, 0], [cols, 0]])
#         matrix = cv2.getPerspectiveTransform(pts1, pts2)
#         distorted = cv2.warpPerspective(img, matrix, (cols, rows))

#         return distorted
