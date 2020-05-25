import dlib
import cv2
from imutils import face_utils
import ourmodulepack as m
from sys import exit
from playsound import playsound
import keyboard
from PIL import Image, ImageDraw, ImageFont
print("프로그램 시작.")
EAR_THRESHOLD = 0.15  # EAR 기준
SLEEPTIME_THRESHOLD = 1.5  # 조는 시간 (단위:초)
FRAMES_PER_SECOND = m.fps_calculate()
COUNTER_THRESHOLD = SLEEPTIME_THRESHOLD * FRAMES_PER_SECOND

strpos1 = (0, 0)
strpos2 = (0, 50)
UsedFont = ImageFont.truetype('/Windows/Fonts/gulim.ttc', 20)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facedetector = dlib.get_frontal_face_detector()
counter = 0
SleepWarning = 0
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    exit("카메라가 감지되지 않았습니다!")
print('감지 시작.')
while True:
    # 1-1
    unused, image = camera.read()
    draw = ImageDraw.Draw(image)
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
        draw.text(strpos1, 'EAR Value: %.3f'.format(average_ear) , fill='green',font=UsedFont)
        if average_ear <= EAR_THRESHOLD and counter < COUNTER_THRESHOLD:
            counter += 1
        elif counter > 0:
            counter -= 1
        SleepWarning = int(counter*3/COUNTER_THRESHOLD)
        if SleepWarning == 3:
            draw.text(strpos2, '졸음 경고!!'.format(average_ear), fill='red', font=UsedFont)
            playsound('alarm.mp3')
        elif SleepWarning > 0:
            draw.text(strpos2, '졸음 주의 %d'.format(SleepWarning), fill='orange', font=UsedFont)
    else: draw.text(strpos1, '눈 감지 실패'.format(average_ear), fill='green', font=UsedFont)
    cv2.imshow('image', image)
    key = cv2.waitKey(1)
    if keyboard.is_pressed('q'):
        break
cv2.destroyAllWindows()
camera.release()
