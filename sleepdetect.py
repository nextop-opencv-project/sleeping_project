from scipy.spatial import distance as dist
import dlib
import cv2
from imutils import face_utils
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector=dlib.get_frontal_face_detector()
#frame=cv2.VideoCapture(0)


#Eye Aspect Ratio: 눈의 모양을 보고 눈이 얼마나 감겨있는지 측정한다
def EAR(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])
    ear=(A+B)/(2.0*C)
    return ear

#임시로 사람 얼굴 이미지를 사용함
image = cv2.imread("human.jpg", cv2.IMREAD_GRAYSCALE)
rects = detector(image,0)
for(i, rect) in enumerate(rects):

    shape = predictor(image,rect)
    shape = face_utils.shape_to_np(shape)
    #얼굴의 특징점 찾기
    print(shape[36:42])
    ear_left = EAR(shape[36:42])
    ear_right = EAR(shape[42:48])
    print(ear_left)
    print(ear_right)