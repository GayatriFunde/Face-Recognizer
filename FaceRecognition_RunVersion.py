# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:12:21 2019

@author: Gayatri
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#os.chdir('F:\Internship_CV\OpenCV_Use\Face_Recognizer')

subjects = ["", "Narendra Modi", "Donald Trumph","Modi with Trumph"]


def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#load OpenCV face detector, I am using LBP which is fast
#there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('F:\Internship_CV\OpenCV_Use\Face_Recognizer\haarcascade_frontalface_default.xml') 
#let's detect multiscale images(some images may be closer to camera than others)
#result is a list of faces
    faces_list = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
#if no faces are detected then return original img
    
#    for (x, y, w, h) in faces:
#        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
#    return gray
    if (len(faces_list) == 0):
        return None, None, None
#under the assumption that there will be only one face,
#extract the face area
#    (x, y, w, h) = faces[0]
    for (x, y, w, h) in faces_list:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = gray[y:y+h,x:x+w]
    print(faces_list)
#return only the face part of the image
    return gray, roi, faces_list


def prepare_training_data(data_folder_path):
    #------STEP-1--------
#get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
 
#list to hold all subject faces
    faces = []
#list to hold labels for all subjects
    labels = []
 
#let's go through each directory and read images within it
    for dir_name in dirs:
 
#our subject directories start with letter 's' so
#ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
 
#------STEP-2--------
#extract label number of subject from dir_name
#format of dir name = slabel
#, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
 
#build path of directory containing images for current subject subject
#sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
 
#get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

#------STEP-3--------
#go through each image name, read image, 
#detect face and add face to list of faces
        for image_name in subject_images_names:
 
#ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
 
#build image path
#sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

#read image
            image = cv2.imread(image_path)
 
#display an image window to show the image 
#            cv2.imshow("Training on image "+image_name, image)
#            cv2.waitKey(10)
 
#detect face
            face, roi, rect = detect_face(image)
            if face is None:
#                plt.imshow(image)
                cv2.imshow(image_name,image)
                cv2.waitKey(1000)
#------STEP-4--------
#for the purpose of this tutorial
#we will ignore faces that are not detected
            if face is not None:
#add face to list of faces
                faces.append(face)
#add label for this face
                labels.append(label)
 
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
 
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("F:\\Internship_CV\\OpenCV_Use\\Face_Recognizer\\training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

#####################
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer = cv2.face.createEigenFaceRecognizer()

face_recognizer.train(faces, np.array(labels))

#function to draw rectangle on image 
def draw_rectangle(img, rect):
#    (x, y, w, h) = rect
    for (x, y, w, h) in rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
def predict(test_img):
#make a copy of the image as we don't want to change original image
    img = test_img.copy()
    img.shape
#detect face from the image
    face, roi, rect = detect_face(img)

#predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
#get name of respective label returned by face recognizer
    label_text = subjects[label]
 
#draw a rectangle around face detected
    draw_rectangle(img, rect)
#draw name of predicted person
    draw_text(img, label_text, rect[0][0], rect[0][1])
 
    return img, label

print("Predicting images...")

#image2 = cv2.imread('F:\Internship_CV\OpenCV_Use\Face2.jpg')
#image2.shape
test_img1 = cv2.imread('F:\Internship_CV\OpenCV_Use\Face_Recognizer\I14.jpg')
test_img2 = cv2.imread('F:\Internship_CV\OpenCV_Use\Face_Recognizer\I12.jpg')
test_img3 = cv2.imread('F:\Internship_CV\OpenCV_Use\Face_Recognizer\I13.jpg')

test_img1.shape
test_img2.shape
test_img3.shape
#plt.imshow(test_img1)
#perform a prediction
predicted_img1,label_id1 = predict(test_img1)
predicted_img2,label_id2 = predict(test_img2)
predicted_img3,label_id3 = predict(test_img3)
print("Prediction complete")

#display both images
cv2.imshow(subjects[label_id1]+'_1', cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[label_id2]+'_2', cv2.resize(predicted_img2, (400, 500)))
cv2.imshow(subjects[label_id3]+'_3', cv2.resize(predicted_img3, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(2)
cv2.destroyAllWindows()

###################
img =cv2.imread('F:\Internship_CV\OpenCV_Use\Face_Recognizer\I13.jpg')
gray, roi, faces = detect_face(img)

print(len(faces))
cv2.imshow('modi',gray)