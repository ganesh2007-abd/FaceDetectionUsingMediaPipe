import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(r"C:\Users\HP\Downloads\5_facedetecting.mp4")
ptime = 0

mpFaceDetection = mp.solutions.face_detection
facedetection = mpFaceDetection.FaceDetection(model_selection=1,min_detection_confidence=0.5)

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
            ih,iw,ic = img.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = (int(bboxC.xmin * iw),int(bboxC.ymin * ih),int(bboxC.width * iw),int(bboxC.height * ih))
            x,y,w,h = bbox
            # print(id,detection)
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            score = int(detection.score[0] * 100)
            cv.putText(img,str(score),(bbox[0],bbox[1]-30),cv.FONT_HERSHEY_COMPLEX,0.75,(0,255,0),2)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,str(int(fps)),(20,70),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv.imshow("Image",img)
    if cv.waitKey(1) == ord('q'):
        print("You interrupted")
        break

cap.release()
cv.destroyAllWindows()