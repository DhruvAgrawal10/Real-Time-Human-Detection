import cv2 as cv
import numpy as np
import tensorflow as tf
import math
import tensorflow_hub as hub

# To load SSD Mobilenet V2
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")

#To load EfficientDet Lite0
# detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1")

#Class labels of COCO dataset used in SSD Mobilenet V2
COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 45: "wine glass", 46: "cup", 47: "fork", 48: "knife", 49: "spoon",
    50: "bowl", 51: "banana", 52: "apple", 53: "sandwich", 54: "orange", 55: "broccoli",
    56: "carrot", 57: "hot dog", 58: "pizza", 59: "donut", 60: "cake", 61: "chair",
    62: "couch", 63: "potted plant", 64: "bed", 65: "dining table", 67: "toilet",
    70: "tv", 72: "laptop", 73: "mouse", 74: "remote", 75: "keyboard", 76: "cell phone",
    77: "microwave", 78: "oven", 79: "toaster", 80: "sink", 81: "refrigerator",
    83: "book", 84: "clock", 85: "vase", 86: "scissors", 87: "teddy bear",
    88: "hair drier", 89: "toothbrush"
}


def preprocess(image):
    image = cv.resize(image, (320, 320))
    return tf.convert_to_tensor(image[np.newaxis, ...])

def draw_boxes(image, boxes, scores, classes, threshold=0.5):
    count = 0
    h, w, _ = image.shape
    for i in range(len(scores)):
        
        if scores[i] > threshold and int(classes[i]) == 1: 
            class_id=int(classes[i])
            class_name = COCO_CLASSES.get(class_id, "Unknown")
            detection_label_full = class_name + ' ' + str(math.floor(100 * scores[i])) + '%'
            y_min, x_min, y_max, x_max = boxes[i]
            x1, y1, x2, y2 = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
            cv.rectangle(image, (x1+10, y1+10), (x2-10, y2-10), (0, 255, 0), 2)
            
            # confidence = f"Conf: {scores[i]:.2f}"
            label_size = cv.getTextSize(
                detection_label_full,
                cv.FONT_HERSHEY_COMPLEX,
                0.7,
                2
            )
            cv.rectangle(image, (x1, y1 - label_size[0][1] - 10), (x1 + label_size[0][0], y1), (0, 255, 0), -1)
            cv.putText(image, detection_label_full, (x1+10, y1 - 10),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,cv.LINE_AA)
            
            count += 1
    return image, count

def rescaleFrame(frame,scale=0.4):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dim=(width,height)
    return cv.resize(frame,dim,interpolation=cv.INTER_AREA)

# Start webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("[INFO] Starting live feed. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break
    
    # Resize frame if needed
    # frame=rescaleFrame(frame,0.3)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    tensor = preprocess(rgb)

    # Use the detector to get boxes, class ids, and scores in SSD Mobilenet V2
    result = detector(tensor)
    result = {k: v.numpy() for k, v in result.items()}
    
    # Use the detector to get boxes, class ids, and scores in efficientdet
    # output = detector(tensor)
    # boxes = output[0].numpy()[0]
    # class_ids = output[1].numpy()[0].astype(np.int32)
    # scores = output[2].numpy()[0]
    
    # Frame in SSD Mobilenet V2
    frame, count = draw_boxes(frame,
                              result["detection_boxes"][0],
                              result["detection_scores"][0],
                              result["detection_classes"][0])
    
    # Used with EfficientDet Lite0
    # frame, count = draw_boxes(frame,
    #                           boxes,
    #                           scores,
    #                           class_ids)

    cv.putText(frame, f'People Detected: {count}', (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
    
    cv.imshow("People Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
