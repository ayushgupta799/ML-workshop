import numpy as np
import imutils
import time
import cv2
import os

iconfidence = 0.5
ithreshold = 0.3

labelsPath = os.path.sep.join(["yolo-coco","coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0,255,size=(len(LABELS),3),dtype="uint8")

weightsPath = os.path.sep.join(["yolo-coco","yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco","yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)

ln = net.getLayerNames()

ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

vc = cv2.VideoCapture(0)
(W,H)= (None,None)
if imutils.is_cv2():
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT

while True:
    (grabbed,frame) = vc.read()

    if not grabbed:
        break
    if W is None or H is None:
        (H,W)=frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    start=time.time()
    layerOutputs=net.forward(ln)
    end=time.time()
 
    boxes=[]
    confidences=[]
    classIDs=[]

    for output in layerOutputs:
        for detection in output:

            scores = detection[:5]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > iconfidence:
                box=detection[0:4]*np.array([W,H,W,H])
                (centerX,centerY,width,height)=box.astype('int')

                x=int(centerX-(width/2))
                y=int(centerY-(height/2))

                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs=cv2.dnn.NMSBoxes(boxes,confidences,iconfidence,ithreshold)

    if len(idxs)>0:
        for i in idxs.flatten():
            (x,y)=(boxes[i][0],boxes[i][1])
            (w,h)=(boxes[i][2],boxes[i][3])

            color=[int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            text="{}:{:.4f}".format(LABELS[classIDs[i]],confidences[i])
            cv2.putText(frame,text,(x,y-5),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,color,2)
    cv2.imshow("My App",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
