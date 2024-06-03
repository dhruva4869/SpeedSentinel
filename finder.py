import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import ObjectTracker  
import time
from math import dist


model = YOLO('yolov8s.pt')

# init
def display_mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        coordinates = [x, y]
        print(coordinates)
        
cv2.namedWindow('Speed Detector')
cv2.setMouseCallback('Speed Detector', display_mouse_position)

video_capture = cv2.VideoCapture('vehicles.mp4')

with open("objects.txt", "r") as file:
    class_names = file.read().split("\n")

# count frames and track object + dimensions
frame_count = 0
tracker = ObjectTracker()
line_y1 = 322
line_y2 = 368

# Offset for line detection
# defines a margin or tolerance range for detecting whether a vehicle's center point crosses a specified line. 
# This offset allows the code to account for minor variations in the exact y-coordinate of the vehicle's center 
# point when determining if it has crossed a line.
line_offset = 6


vehicles_down = {}
vehicles_up = {}
counted_down = []
counted_up = []

while True:    
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # check each third frame
    frame_count += 1
    if frame_count % 3 != 0:
        continue
    
    frame = cv2.resize(frame, (1057, 523))
    
    # object detection
    results = model.predict(frame)
    detection_data = results[0].boxes.data
    detection_df = pd.DataFrame(detection_data).astype("float")
    
    # save bounding boxes for cars
    detected_cars = []

    for _, row in detection_df.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row)
        class_name = class_names[class_id]
        if 'car' in class_name:
            detected_cars.append([x1, y1, x2, y2])
    
    tracked_objects = tracker.update(detected_cars)
    
    for bbox in tracked_objects:
        x1, y1, x2, y2, object_id = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # make bb
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # coming towards screen + find speed
        if line_y1 - line_offset < center_y < line_y1 + line_offset:
            vehicles_down[object_id] = time.time()
        
        if object_id in vehicles_down:
            if line_y2 - line_offset < center_y < line_y2 + line_offset:
                elapsed_time = time.time() - vehicles_down[object_id] # currtime-prev stored time
                if object_id not in counted_down:
                    counted_down.append(object_id)
                    distance = 10
                    speed_mps = distance / elapsed_time
                    speed_kph = speed_mps * 3.6
                    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(object_id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255,255), 1)
                    cv2.putText(frame, f'{int(speed_kph)} Km/h', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    if speed_kph > 12:
                        flag_position = (x2 + 10, y2 - 10)
                        cv2.putText(frame, 'FLAG', flag_position, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        # going away
        if line_y2 - line_offset < center_y < line_y2 + line_offset:
            vehicles_up[object_id] = time.time()
        
        if object_id in vehicles_up:
            if line_y1 - line_offset < center_y < line_y1 + line_offset:
                elapsed_time = time.time() - vehicles_up[object_id]
                if object_id not in counted_up:
                    counted_up.append(object_id)
                    distance = 10
                    speed_mps = distance / elapsed_time
                    speed_kph = speed_mps * 3.6
                    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(object_id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f'{int(speed_kph)} Km/h', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    if speed_kph > 12:
                        flag_position = (x2 + 10, y2 - 10)
                        cv2.putText(frame, 'FLAG', flag_position, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.line(frame, (274, line_y1), (814, line_y1), (255, 255, 255), 1)
    cv2.putText(frame, 'L1', (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (177, line_y2), (927, line_y2), (255, 255, 255), 1)
    cv2.putText(frame, 'L2', (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    
    down_count = len(counted_down)
    up_count = len(counted_up)
    cv2.putText(frame, f'Coming Towards: {down_count}', (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Going Away: {up_count}', (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    
    # esc key for exit
    cv2.imshow("Speed Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

video_capture.release()
cv2.destroyAllWindows()
