import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION
print(cv2.__version__)
print(np.__version__)

Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush' ]

Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('weights/yolov5s.pt', Object_classes)


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):

    return ('v4l2src device=/dev/video{} ! '
            'video/x-raw, width=(int){}, height=(int){} ! '
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            ).format(0, capture_width, capture_height)


dispW=640
dispH=480
flip=2
onboardCamera = ('nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, '
                'height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! '
                'video/x-raw, width='+str(dispW)+', height='+str(dispH)+','
                ' format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink')

usbCamera = ('v4l2src device=/dev/video{} ! '
            'video/x-raw, width=(int){}, height=(int){} ! '
            'videoconvert ! appsink').format(0, 640, 480)

rtspCamera = ('rtspsrc location={} latency={} ! '
            'rtph264depay ! h264parse ! omxh264dec ! '
            'nvvidconv ! '
            'video/x-raw, width=(int){}, height=(int){}, '
            'format=(string)BGRx ! '
            'videoconvert ! appsink').format('rtsp://jetson:jetson@10.0.2.203/live', 10, 320, 240)

cam=cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

if cam.isOpened():
    window_handle1 = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
    window_handle2 = cv2.namedWindow("Detector", cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame=cam.read()
    if ret:
        objs = Object_detector.detect(frame)
            # plotting
        for obj in objs:
            # print(obj)
            label = obj['label']
            score = obj['score']
            [(xmin,ymin),(xmax,ymax)] = obj['bbox']
            color = Object_colors[Object_classes.index(label)]
            frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
            frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)
        cv2.imshow('Camera', frame)

    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()