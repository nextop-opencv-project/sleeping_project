import dlib
from imutils import face_utils
from ourmodulepack import *
from sys import exit
import winsound
import keyboard
from datetime import datetime
print("프로그램 시작.")
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    exit("카메라가 감지되지 않았습니다!")
SLEEPTIME_THRESHOLD = 1.5  # 조는 시간 (단위:초)
SleepWarning = 0
counter = 0
orange = (0, 127, 255)
red = (0, 0, 255)
green = (0, 255, 0)
strpos1 = (0, 20)
strpos2 = (0, 50)
UsedFont = cv2.FONT_HERSHEY_PLAIN
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
facedetector = dlib.get_frontal_face_detector()
FRAMES_PER_SECOND = fps_calculate()
COUNTER_THRESHOLD = SLEEPTIME_THRESHOLD * FRAMES_PER_SECOND
while True:
    lv = int(input('감지 민감도를 선택하세요.(1~4단계)'))
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
        print('1~4 사이의 숫자를 입력하세요.')
print('감지 시작.')
Starttime = datetime.today()
SleepTime = datetime.today()
Sleeping = False
while True:
    # 1-1
    unused, image = camera.read()
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 1-2?
    haardetected = False
    d = facecascade.detectMultiScale(grayimg, 1.5, 5)
    for x, y, w, h in d:
        haardetected = True
        break
    rects = facedetector(grayimg, 0)
    average_ear = 0
    for (i, rect) in enumerate(rects):
        shape = predictor(grayimg, rect)
        shape = face_utils.shape_to_np(shape)
        # 2-1은 2-2를 하는 과정에서 자연스럽게 되므로 스킵
        # 2-2
        Lefteye = shape[36:42]
        Righteye = shape[42:48]
        ear_left = EAR(Lefteye)  # 왼쪽눈
        ear_right = EAR(Righteye)  # 오른쪽눈
        average_ear = (ear_left + ear_right) / 2
        Lefthull = cv2.convexHull(Lefteye)
        Righthull = cv2.convexHull(Righteye)
        cv2.drawContours(image, [Lefthull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [Righthull], -1, (0, 255, 0), 1)

    if average_ear == 0:  # 눈 감지 안됨
        cv2.putText(image, 'Eye not detected.', strpos1, UsedFont, 2, green, 2)
    else:
        cv2.putText(image, 'EAR Value: {:.3}'.format(average_ear), strpos1, UsedFont, 2, green, 2)
    if average_ear <= EAR_THRESHOLD and counter <= COUNTER_THRESHOLD:
        counter += 1
    elif counter > 0:
        counter -= 1
    SleepWarning = int(counter * 3 / COUNTER_THRESHOLD)
    if SleepWarning > 0:
        if not Sleeping:
            SleepTime = datetime.today()
            Sleeping = True
        if SleepWarning == 3:
            cv2.putText(image, 'SLEEPING ALERT!', strpos2, UsedFont, 2, red, 2)
            winsound.PlaySound('alarm.wav', winsound.SND_FILENAME)
        else: cv2.putText(image, 'Sleeping warning lv {}'.format(SleepWarning), strpos2, UsedFont, 2, orange, 2)
    elif Sleeping:
        Starttime += SleepTime-datetime.today()
        Sleeping = False
    runtime = datetime.today()-Starttime
    cv2.putText(image, '{}'.format(runtime), (0, 80), UsedFont, 2, orange, 2)
    cv2.imshow('screen', image)
    key = cv2.waitKey(1)
    if keyboard.is_pressed('q'):
        break
cv2.destroyAllWindows()
camera.release()
