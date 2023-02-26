# face_recog_git
Hello

This face recognition project

Using this code you can identify the person who was added to the database

# USED LIBRARIES

openCV

face-recognition

numpy

datetime

os

click

pyfiglet    

# INSTALLING

pip install -r requirements.txt

python3 realease.py

# USAGE

1. choose index of camera
    if you dont know nothing about your cam index DONT TRY and press ENTER
2. to save unknown face press "s" and write info
    faces saving in folder /faces
    filename format: Sname_Name_datatime_<.file extension>  datatime - 31 symbols
3. if error about index of array delete last saved face from folder

# FACE BASE

Adding face:
1. Crop the image so that only the part with the face remains
2. Move image to folder faces/
3. Rename the image using format
    <SName>_<Name>_<31 symbols of data time><.file extension>
    data time format yyyy-mm-dd hh:mm:ss.<6 symbols of frame number(can use random)>

# CREATED BY Bizon and Captain Tab
