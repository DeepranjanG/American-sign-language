import cv2
import time


def predict(filename):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)

    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 0), (255, 0, 0)]

    class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    frame = cv2.imread(filename)

    net = cv2.dnn.readNet("ASL-best.weights", "ASL.cfg")


    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)


    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # cv2.imshow("detections", cv2.resize(frame, (1280, 640)))
    cv2.imwrite(f"static/uploads/output.png", frame)
    # cv2.waitKey(0)