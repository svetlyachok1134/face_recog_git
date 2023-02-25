import face_recognition
import cv2
import numpy as np
import datetime
import os
import click
import pyfiglet

face = []
path= '/faces'
lastFrame = []
face_locations = []
face_encodings = []
face_names = []
cam_index = 0
# Get a reference to webcam #0 (the default one)

recognizer_cc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

###############################################

def welcome():
    os.system("mode con cols=00 lines=50")
    ascii_banner = pyfiglet.figlet_format("Hello!!")
    print(ascii_banner)
    click.pause('press any key to continue')
    os.system('cls||clear')

    ascii_banner = pyfiglet.figlet_format("This program was created by Bizon")
    print(ascii_banner)
    click.pause('press any key to continue')
    os.system('cls||clear')

    ascii_banner = pyfiglet.figlet_format("captain tab")
    print(ascii_banner)

    print('2023')

    click.pause('press any key to continue')
    os.system('cls||clear')

    cam_index = input('choose index of your camera (0 - integrated web cam)')

    print('to save face press "s"')
    print('to exit rpess "q"')
    print('to clear screan pres "c')


video_capture = cv2.VideoCapture(cam_index)

dir = 'faces/'
known_face_encodings = []
known_face_names =[]
photosEnc = []
#listing the face/ directory and loading faces with names
def fileWork():

    files = os.listdir(dir)
    for element in files :   
        personname = element[:-31]
        photo = face_recognition.load_image_file(dir + element)
        if len(face_recognition.face_encodings(photo)) != 0:
            photoEnc = face_recognition.face_encodings(photo)[0]
            known_face_encodings.append(photoEnc)

           
        #print(personname)
            forEnc = personname + '_enc'
            known_face_names.append(personname)
        else :
            print(personname +' face not detected' )
            #os.remove(element)


##############################################
# Initialize some variables

process_this_frame = True

#saving new faces
def imgSave(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
gray,
scaleFactor=1.2,
minNeighbors=2,
minSize=(10, 10)
)
    for (x, y, w, h) in faces:
        out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = gray[y:y+h,x:x+w]
    #cv2.imshow('output' , face)
    name = input('write the name\n')
    sname = input('write the sname\n')
    fullname = sname + '_' + name + '_' + str(curTime)
    fullfilename = dir + fullname + '.jpg'
    if len(face) != 0:
        cv2.imwrite(fullfilename, face) 



fileWork()
print('to escape press "q"\n')
welcome()
while True:
    curTime = datetime.datetime.now()
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]                  

            if name == 'Unknown' :
                imgText = ' to save face press S'
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    if len(frame) > 0:
                        imgSave(frame)
                    else :
                        print("bad frame, tyr again")
                    fileWork()

            else :
                imgText = ' '
            face_names.append(name)

    process_this_frame = not process_this_frame

    lastFrame = frame
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, str(curTime), (0,25), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, imgText, (0,150), font, 1.0, (255, 255, 255), 1)
       # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        os.system('cls || clear')

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
