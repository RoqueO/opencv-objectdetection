import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from screeninfo import get_monitors
from ultralytics import YOLO

def createwindows(window_names: list, width=400, height=300):
    for i, window_name in enumerate(window_names):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        x = i % 2 * width
        y = i // 2 * height
        cv2.moveWindow(window_name, x, y)
        #cv2.resizeWindow(window_name, width, height)

def runclassicalmethod(capture):
    backSub = cv2.createBackgroundSubtractorMOG2()
    if not cap.isOpened():
        print("Error opening video file")

    createwindows(window_names=['Subtractor', 'Eroded', 'Processed'], width=window_size[0], height=window_size[1])

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Apply background subtraction
            fg_mask = backSub.apply(frame, fgmask=None)

            # apply global threshold to remove shadows
            retval, mask_thresh = cv2.threshold( fg_mask, 150, 255, cv2.THRESH_BINARY)

            # set the kernal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            # Apply erosion
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(contours)
            #frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            min_contour_area = 1000  # Define your minimum area threshold
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            # Draw bounding boxes
            frame_out = frame.copy()
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)

            # Display the resulting frame
            cv2.imshow('Processed', cv2.resize(frame_out, window_size))
            cv2.imshow('Subtractor', cv2.resize(fg_mask, window_size))
            cv2.imshow('Eroded', cv2.resize(mask_eroded, window_size))
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break

    # When everything done, release 
    # the video capture object 
    cap.release() 
    
    # Closes all the frames 
    cv2.destroyAllWindows() 

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

        annotated_frame = results[0].plot()
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                if box.cls[0] == 15:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    #print("Confidence --->",confidence)

                    # class name
                    #cls = int(box.cls[0])
                    #print("Class name -->", classNames[cls])

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, 'Cat - {confidence}'.format(confidence=confidence), org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', annotated_frame)
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
    #runclassicalmethod(cap)
    #runhaasdetection(cap)
    #runyolov5(cap)
    runyolov8(cap)
    