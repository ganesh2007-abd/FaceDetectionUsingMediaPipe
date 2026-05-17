import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(r"C:\Users\HP\Downloads\7279002-uhd_4096_2160_24fps.mp4")
ptime = 0

mpFaceDetection = mp.solutions.face_detection
facedetection = mpFaceDetection.FaceDetection()

mpDraw = mp.solutions.drawing_utils

while True:
    success,img = cap.read()
    if not success:
        print("Video Ended")
        break
    img = cv.resize(img,None,fx=0.25,fy=0.25,interpolation=cv.INTER_AREA)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = facedetection.process(imgRGB)
    if results.detections:
        for id,detection in enumerate(results.detections):
            mpDraw.draw_detection(img,detection)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,str(int(fps)),(20,70),cv.FONT_HERSHEY_COMPLEX,3,(0,255,0),2)

    cv.imshow("Image",img)
    if cv.waitKey(1) == ord('q'):
        print("You interrupted")
        break

cap.release()
cv.destroyAllWindows()