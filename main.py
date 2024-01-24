import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time


training_path = 'Training_images'


training_images = []
class_names = []


for file_name in os.listdir(training_path):
    current_image = cv2.imread(f'{training_path}/{file_name}')
    training_images.append(current_image)
    class_names.append(os.path.splitext(file_name)[0])


def find_encodings(images_list):
    encodings_list = []

    for image in images_list:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(rgb_image)[0]
        encodings_list.append(face_encoding)

    return encodings_list


def mark_attendance(name):
    with open('Attendance.csv', 'a') as attendance_file: 
        now = datetime.now()
        date_time_string = now.strftime('%Y-%m-%d %H:%M:%S')
        attendance_file.write('\n{},{}'.format(name, date_time_string)) 


known_encodings = find_encodings(training_images)


cap = cv2.VideoCapture(0)


last_detection_time = {}


min_detection_interval = 0 

while True:
    success, webcam_image = cap.read()

   
    resized_image = cv2.resize(webcam_image, (0, 0), None, 0.25, 0.25)
    rgb_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)


    faces_current_frame = face_recognition.face_locations(rgb_resized_image)
    encodings_current_frame = face_recognition.face_encodings(rgb_resized_image, faces_current_frame)

    for encode_face, face_location in zip(encodings_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(known_encodings, encode_face)
        face_distances = face_recognition.face_distance(known_encodings, encode_face)

        match_index = np.argmin(face_distances)

        if matches[match_index]:
            identified_name = class_names[match_index].upper()

      
            current_time = time.time()
            last_detection_time_person = last_detection_time.get(identified_name, 0)
            time_since_last_detection = current_time - last_detection_time_person

            if time_since_last_detection >= min_detection_interval:
                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(webcam_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(webcam_image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(webcam_image, identified_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(identified_name)

               
                last_detection_time[identified_name] = current_time

  
    cv2.imshow('Live Recognition', webcam_image)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


time.sleep(2)
cap.release()
cv2.destroyAllWindows()