import dlib
import cv2
from imutils import face_utils
import ourmodulepack as m
import keyboard
import sys
from playsound import playsound
EAR_THRESHOLD = 0.15  # EAR 기준
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
    # 1-1
    image = camera.read()
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 1-2?
    rects = detector(grayimg, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(grayimg, rect)
        shape = face_utils.shape_to_np(shape)
        # 2-1은 2-2를 하는 과정에서 자연스럽게 되므로 스킵
        # 2-2
        Lefteye=shape[36:42]
        Righteye=shape[42:48]
        ear_left = m.EAR(Lefteye)  # 왼쪽눈
        ear_right = m.EAR(Righteye)  # 오른쪽눈
        average_ear = (ear_left + ear_right) / 2
        # 3
        if average_ear <= EAR_THRESHOLD:
            counter += 1
        elif counter > 0:
            counter -= 1
        Lefthull = cv2.convexHull(Lefteye)
        Righthull = cv2.convexHull(Righteye)
        cv2.drawContours(image, [Lefthull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [Righthull], -1, (0, 255, 0), 1)
        if counter >= COUNTER_THRESHOLD:
            sleeping = True
        else:
            sleeping = False
        if sleeping:
            print("졸음 경고!!")
            playsound('alarm.mp3')
    cv2.imshow('facecam',image)
    if keyboard.is_pressed('q'):  # 'q'를 누르면 종료
        break
cv2.destroyAllWindows()
camera.release()
