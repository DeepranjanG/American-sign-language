import cv2
import time
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 0)]



class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

vc = cv2.VideoCapture("asl-alphabet.mp4")

net = cv2.dnn.readNet("ASL-best.weights", "ASL.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'results/Output_{current_time}.avi', fourcc, 30, (int(vc.get(3)),int(vc.get(4))))

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        print(classid)
        color = COLORS[int(classid) % len(COLORS)]
        print(color)
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    out.write(frame)
    cv2.imshow("detections", cv2.resize(frame, (1080, 640)))