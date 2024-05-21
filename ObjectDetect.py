import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from methods import classical
from screeninfo import get_monitors
from ultralytics import YOLO

track_history = defaultdict(lambda: [])

def createwindows(window_names: list, width=400, height=300):
    for i, window_name in enumerate(window_names):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        x = i % 2 * width
        y = i // 2 * height
        cv2.moveWindow(window_name, x, y)
        #cv2.resizeWindow(window_name, width, height)

def runhaasdetection(cap):
    # Load the cascade
    cat_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalcatface_extended.xml')

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cats
        cats = cat_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,    # Adjust this value
            minNeighbors=4,     # Adjust this value
            minSize=(75, 75),   # Adjust this value based on the size of your cat's face
            maxSize=(300, 300)  # Adjust this value if necessary
        )

        # Draw rectangles around the cats
        for (x, y, w, h) in cats:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Cat Detector', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def runyolov5(cap):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the image to RGB (YOLO expects RGB images)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(img_rgb, size=640)

        # Parse results
        for detection in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = detection
            if cls == 15:  # Class ID for 'cat' in COCO dataset
                # Draw a rectangle around the detected cat
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Cat Detector', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def runyolov8(cap):
    model = YOLO("yolov8n.pt")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    #img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    while True:
        success, img = cap.read()
        results = model.track(img, persist=True)

        # Get the boxes and track ids
        #boxes_xyxy = results[0].boxes.xyxy.cpu()
        #boxes_xywh = results[0].boxes.xywh.cpu()
        #track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # coordinates
        for r in results:
            for box in boxes:
                if box.cls[0] == 15:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100

                    # tracking
                    x, y, w, h = box.xywh[0]
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    # Draw the tracking line
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, 'Cat - {confidence}'.format(confidence=confidence), org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = get_monitors()[0]
    window_size = (monitor.width // 2, monitor.height // 2)

    deviceID = 0;           # 0 = open default camera
    apiID = cv2.CAP_ANY;    # 0 = autodetect default API

    cap = cv2.VideoCapture(deviceID, apiID)
    runclassicalmethod(cap)
    #runhaasdetection(cap)
    #runyolov5(cap)
    runyolov8(cap)
    