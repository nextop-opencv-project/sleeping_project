import cv2
import time
from scipy.spatial import distance as dist


def fps_calculate():
    video = cv2.VideoCapture(0)
    num_frames = 300
    start = time.time()
    for i in range(0, num_frames):
        ret, frame = video.read()
    end = time.time()
    seconds = end - start
    fps = num_frames / seconds
    video.release()
    return fps

# Eye Aspect Ratio: 눈의 모양을 보고 눈이 얼마나 감겨있는지 측정한다
def EAR(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear