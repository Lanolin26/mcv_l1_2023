import time
import cv2
import numpy as np


def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30,
                       flip_method=0):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
            % (capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )


u_green = np.array([255, 255, 70])
l_green = np.array([30, 30, 0])
display_width_all = 1280
display_height_all = 720


def applied_chroma_key(frame, image):
    mask = cv2.inRange(frame, l_green, u_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    applied_frame = frame - res
    applied_frame = np.where(applied_frame == 0, image, applied_frame)
    return applied_frame


def applied_chroma_key_v2(frame, image):
    a_g = np.greater_equal(frame, l_green)
    a_l = np.less_equal(frame, u_green)
    a = np.logical_and(a_g, a_l)
    mask = np.all(a, axis=2, keepdims=True)
    res = np.bitwise_and(frame, frame, where=mask)
    applied_frame = frame - res
    applied_frame = np.where(applied_frame == 0, image, applied_frame)
    return applied_frame


def start(image_name, video_name='', is_camera=False, algo=1):
    if is_camera:
        gst = gstreamer_pipeline(display_width=display_width_all, display_height=display_height_all, flip_method=4)
        video = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    else:
        video = cv2.VideoCapture(video_name)

    image = cv2.imread(image_name)
    image = cv2.resize(image, (display_width_all, display_height_all))

    frame_time = []

    if video.isOpened():
        cv2.namedWindow("REAL", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("REAL", 0) >= 0:
            ret_val, frame = video.read()
            if not ret_val:
                break

            frame = cv2.resize(frame, (display_width_all, display_height_all))
            cv2.imshow('REAL', frame)

            startTime = time.time_ns()
            if algo == 1:
                applied_frame = applied_chroma_key(frame, image)
            elif algo == 2:
                applied_frame = applied_chroma_key_v2(frame, image)
            else:
                applied_frame = frame
            endTime = time.time_ns()
            howMuchTime = endTime - startTime
            frame_time.append(howMuchTime)

            cv2.imshow('MASKED', applied_frame)

            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break

        video.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
    return frame_time


def performance(algo, times, title, image, video):
    print(title)
    ocv_array = []
    for i in range(times):
        frame_1_time = start(image_name=image, video_name=video, algo=algo)
        frame_1_mean = np.mean(frame_1_time) / 1000
        print(title + " (" + str(i+1) + " attempt): " + str(frame_1_mean) + " μs per frame / FPS: " + str(1000000/frame_1_mean))
        ocv_array.append(frame_1_mean)
    np_mean = np.mean(ocv_array)
    print(title + " * " + str(times) + " times: " + str(np_mean) + " μs per frame;")
    print(title + " Mean FPS: " + str(1000000/np_mean))


def start_perf(image, video):
    performance(1, 5, "OpenCV + NumPy", image, video)
    print()
    performance(2, 5, "NumPy", image, video)


if __name__ == '__main__':
    start_perf("5ccdcd50fe2878225ddb25dca9f1d0c5.jpg", "bulb.mp4")
