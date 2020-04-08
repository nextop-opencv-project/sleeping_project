from scipy.spatial import distance as dist
import dlib
import cv2
from imutils import face_utils
import fpscalculate as f 
import keyboard
import sys
EAR_THRESHOLD = 2.5  # EAR 기준
SLEEPTIME_THRESHOLD = 2  # 조는 시간 (단위:초)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


# Eye Aspect Ratio: 눈의 모양을 보고 눈이 얼마나 감겨있는지 측정한다
def EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


camera = cv2.VideoCapture(0)
if not camera.isOpened():
    sys.exit("카메라가 감지되지 않았습니다!")

counter = 0
sleeping = False
FRAMES_PER_SECOND = f.fps_calculate()
COUNTER_THRESHOLD = SLEEPTIME_THRESHOLD * FRAMES_PER_SECOND
while True:
    image = camera.read()  # <-웹캠에서 지금 순간의 이미지를 가져오는 코드
    # image = cv2.imread("human.jpg", cv2.IMREAD_GRAYSCALE)  # <-웹캠이 없으니 이미지로 대체
    rects = detector(image, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        # 얼굴의 특징점 찾기
        print(shape[36:42])
        ear_left = EAR(shape[36:42])
        ear_right = EAR(shape[42:48])
        average_ear = (ear_left + ear_right) / 2
        if average_ear >= EAR_THRESHOLD:
            counter += 1
        elif counter > 0:
            counter -= 1
        if counter >= COUNTER_THRESHOLD:
            sleeping = True
        else:
            sleeping = False
        if sleeping:
            print("졸음 경고!!")
            # 졸음 감지되었을 때 실행할 코드
        if keyboard.is_pressed('q'):  # 'q'를 누르면 종료
            break
