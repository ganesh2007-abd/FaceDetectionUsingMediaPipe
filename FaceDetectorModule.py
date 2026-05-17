import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self,model=0,MinDetectingCon = 0.5):
        self.model = model
        self.MinDetectingCon = MinDetectingCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.facedetection = self.mpFaceDetection.FaceDetection(model_selection=self.model,min_detection_confidence=self.MinDetectingCon)
                                        
                                        

        self.mpDraw = mp.solutions.drawing_utils


    def FindFace(self,img):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.facedetection.process(imgRGB)
        bboxes =[]
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                ih,iw,ic = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = (int(bboxC.xmin * iw),int(bboxC.ymin * ih),int(bboxC.width * iw),int(bboxC.height * ih))
                x,y,w,h = bbox
                # print(id,detection)
                bboxes.append([id,detection.score,bbox])
                cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                score = int(detection.score[0] * 100)
                cv.putText(img,str(score),(bbox[0],bbox[1]-30),cv.FONT_HERSHEY_COMPLEX,0.75,(0,255,0),2)

        return img,bboxes


def main():
    cap = cv.VideoCapture(r"C:\Users\HP\Downloads\5_facedetecting.mp4")
    # cap = cv.VideoCapture(0)
    ptime = 0
    detector = FaceDetector(1)
    while True:
        success,img = cap.read()
        
        if not success:
            print("Video Ended")
            break
        
        img = cv.resize(img,None,fx=0.25,fy=0.25,interpolation=cv.INTER_AREA)
        img,boxes = detector.FindFace(img)

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

if __name__ == "__main__":
    main()