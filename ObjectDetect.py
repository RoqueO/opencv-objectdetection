import cv2
import numpy as np
import matplotlib.pyplot as plt

def createwindows(window_names: list, width=1366, height=768):
    for i, window_name in enumerate(window_names):``
        cv2.namedWindow(window_name)
        x = i % 2 * width
        y = i // 2 * height
        cv2.moveWindow(window_name, x, y)

if __name__ == "__main__":
    deviceID = 0;           # 0 = open default camera
    apiID = cv2.CAP_ANY;    # 0 = autodetect default API

    cap = cv2.VideoCapture(deviceID, apiID)
    backSub = cv2.createBackgroundSubtractorMOG2()
    if not cap.isOpened():
        print("Error opening video file")

    createwindows(window_names=['Subtractor', 'Eroded', 'Processed'])

    while cap.isOpened():
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Apply background subtraction
                fg_mask = backSub.apply(frame, fgmask=None, learningRate=0.2)

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
                cv2.imshow('Processed', frame_out)
                cv2.imshow('Subtractor', fg_mask)
                cv2.imshow('Eroded', mask_eroded)
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    break

    # When everything done, release 
    # the video capture object 
    cap.release() 
    
    # Closes all the frames 
    cv2.destroyAllWindows() 