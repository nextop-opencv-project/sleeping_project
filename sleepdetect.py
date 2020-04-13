import dlib
import cv2
from imutils import face_utils
import ourmodulepack as m
import keyboard
import sys
from playsound import playsound
EAR_THRESHOLD = 2.5  # EAR 기준
SLEEPTIME_THRESHOLD = 2  # 조는 시간 (단위:초)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
counter = 0
sleeping = False
FRAMES_PER_SECOND = m.fps_calculate()
COUNTER_THRESHOLD = SLEEPTIME_THRESHOLD * FRAMES_PER_SECOND

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    sys.exit("카메라가 감지되지 않았습니다!")


while True:
    image = camera.read()  # <-웹캠에서 지금 순간의 이미지를 가져오는 코드
    # image = cv2.imread("human.jpg", cv2.IMREAD_GRAYSCALE)  # <-웹캠이 없으니 이미지로 대체
    rects = detector(image, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        # 얼굴의 특징점 찾기
        print(shape[36:42])
        # 두 눈의 EAR 값을 계산, 평균값을 구한다.
        ear_left = m.EAR(shape[36:42])
        ear_right = m.EAR(shape[42:48])
        average_ear = (ear_left + ear_right) / 2
        # EAR 값이 기준보다 높으면 카운터 증가
        if average_ear >= EAR_THRESHOLD:
            counter += 1
        elif counter > 0:
            counter -= 1
        # 카운터가 기준값을 넘으면 경고 출력
        if counter >= COUNTER_THRESHOLD:
            sleeping = True
        else:
            sleeping = False
        if sleeping:
            print("졸음 경고!!")
            playsound('alarm.mp3')
        if keyboard.is_pressed('q'):  # 'q'를 누르면 종료
            break
