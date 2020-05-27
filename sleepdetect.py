import dlib
import cv2
from imutils import face_utils
import ourmodulepack as m
from sys import exit
from playsound import playsound
import keyboard

print("프로그램 시작.")
SLEEPTIME_THRESHOLD = 1.5  # 조는 시간 (단위:초)
FRAMES_PER_SECOND = m.fps_calculate()
COUNTER_THRESHOLD = SLEEPTIME_THRESHOLD * FRAMES_PER_SECOND
counter = 0
orange = (0, 127, 255)
red = (0, 0, 255)
green = (0, 255, 0)
strpos1 = (0, 20)
strpos2 = (0, 50)
UsedFont = cv2.FONT_HERSHEY_PLAIN
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facedetector = dlib.get_frontal_face_detector()
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    exit("카메라가 감지되지 않았습니다!")
while True:
    lv = input('감지 민감도를 선택하세요.(1~4단계)')
    if lv == 1:
        EAR_THRESHOLD = 0.1
        break
    elif lv == 2:
        EAR_THRESHOLD = 0.12
        break
    elif lv == 3:
        EAR_THRESHOLD = 0.15
        break
    elif lv == 4:
        EAR_THRESHOLD = 0.2
        break
    else:
        print('1~4 사이이 숫자를 입력하세요.')
print('감지 시작.')
while True:
    # 1-1
    unused, image = camera.read()
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 1-2?
    rects = facedetector(grayimg, 0)
    eyedetected = False
    average_ear = 0
    for (i, rect) in enumerate(rects):
        eyedetected = True
        shape = predictor(grayimg, rect)
        shape = face_utils.shape_to_np(shape)
        # 2-1은 2-2를 하는 과정에서 자연스럽게 되므로 스킵
        # 2-2
        Lefteye = shape[36:42]
        Righteye = shape[42:48]
        ear_left = m.EAR(Lefteye)  # 왼쪽눈
        ear_right = m.EAR(Righteye)  # 오른쪽눈
        average_ear = (ear_left + ear_right) / 2
        Lefthull = cv2.convexHull(Lefteye)
        Righthull = cv2.convexHull(Righteye)
        cv2.drawContours(image, [Lefthull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [Righthull], -1, (0, 255, 0), 1)
    if eyedetected:
        cv2.putText(image, 'EAR Value: {:.3}'.format(average_ear), strpos1, UsedFont, 1, green, 2)
        if average_ear <= EAR_THRESHOLD and counter < COUNTER_THRESHOLD:
            counter += 1
        elif counter > 0:
            counter -= 1
        SleepWarning = int(counter * 3 / COUNTER_THRESHOLD)
        if SleepWarning == 3:
            cv2.putText(image, 'SLEEPING ALERT!', strpos2, UsedFont, 2, red, 2)
            playsound('alarm.mp3')
        elif SleepWarning > 0:
            cv2.putText(image, 'Sleeping warning lv {}'.format(SleepWarning), strpos2, UsedFont, 2, orange, 2)
    else:
        cv2.putText(image, 'Eye not detected.', strpos1, UsedFont, 2, green, 2)
    cv2.imshow('screen', image)
    key = cv2.waitKey(1)
    if keyboard.is_pressed('q'):
        break
cv2.destroyAllWindows()
camera.release()
