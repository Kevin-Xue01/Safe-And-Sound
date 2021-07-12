import cv2


class Util:
    def freeze_current_image_array(images, titles):
        for index in range(len(images)):
            cv2.imshow(titles[index], images[index])
        while True:
            if cv2.waitKey(1) == ord("q"):  # press q to terminate program
                break
        cv2.destroyAllWindows()

    def freeze_current_image(image, title="Untitled"):
        cv2.imshow(title, image)
        while True:
            if cv2.waitKey(1) == ord("q"):  # press q to terminate program
                break
        cv2.destroyAllWindows()


def start_recording_video():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        (
            ret,
            frame,
        ) = cap.read()
        # mask = apply_green_mask(frame)
        # contours = find_green_rectangle_contours(mask)
        # find_green_rectangle_polygon(contours, frame)
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) == ord("q"):  # press q to terminate program
            break

    cap.release()
    cv2.destroyAllWindows()
