import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture= cv2.VideoCapture(0)

sameer_image = face_recognition.load_image_file("faces/sameer.jpg")
sameer_encoding = face_recognition.face_encodings(sameer_image)[0]

bhanu_image = face_recognition.load_image_file("faces/bhanu.jpg")
bhanu_encoding = face_recognition.face_encodings(bhanu_image)[0]

known_face_encodings = [sameer_encoding,bhanu_encoding]
known_face_names = ["sameer","bhanu"]

students = known_face_names.copy()

face_locations =[]
face_encodings =[]


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f=open(f"{current_date}.csv", "w+" , newline="")
lnwriter = csv.writer(f)

while True:
     _, frame = video_capture.read()
     small_frame = cv2.resize(frame,(0,0),fx=0.25, fy=0.25)
     rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)


     #Recoginition faces
     face_locations = face_recognition.face_locations(rgb_small_frame)
     face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)


     for face_encoding in face_encodings:
          mathes = face_recognition.compare_faces(known_face_encodings,face_encoding)
          face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
          best_match_index = np.argmin(face_distance)

          if(mathes[best_match_index]):
              name = known_face_names[best_match_index]
          if name in known_face_names:
               font = cv2.FONT_HERSHEY_SIMPLEX
               bootomLeftCornerOfText = (10,100)
               fontscale = 1.5
               fontColor = (255,0,0)
               thickness = 3
               lineType = 2
               cv2.putText (frame,name + "Present", bootomLeftCornerOfText,font,fontscale,fontColor,thickness,lineType, )




               if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M%S")
                    lnwriter.writerow([name,current_time])


     cv2.imshow("Attendance",frame)
     if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyWindow()
f.close()











