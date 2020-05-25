import cv2
import time
from scipy.spatial import distance as dist


def fps_calculate():
    print('프레임 레이트 계산 시작.')
    video = cv2.VideoCapture(0)
    num_frames = 100
    start = time.time()
    for i in range(0, num_frames):
        ret, frame = video.read()
        if i % 25 == 0:
            print('프레임 레이트 계산중...%d%'.format(i))
    end = time.time()
    seconds = end - start
    if seconds == 0:
        print('프레임 레이트 계산 중 오류 발생. 기본값인 60으로 설정함.')
        return 60
    fps = num_frames / seconds
    video.release()
    print('프레임 레이트 계산 완료. FPS: ', fps)
    return fps


# Eye Aspect Ratio: 눈의 특징점 6개의 좌표로 눈이 얼마나 감겨있는지
def EAR(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear
