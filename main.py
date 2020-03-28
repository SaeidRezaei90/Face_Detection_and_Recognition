#import opencv module
import cv2
#import os module for reading data directories and paths
import os
#import numpy to convert python lists to numpy arrays as it is needed by openCV file recognation
import numpy as np

#there is no lable 0 in pur training data so subject name for index 0 is empt
subjects = ["","Ronaldo","Messi","Robben"]

#Function to detect face using openCV
def detect_face(img):
    #Convert the image to a gray image as openCV face detector expects gray image
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #load opencv face detector usin LBP which is fast
    face_cascade = cv2.CascadeClassifier("opencv-files/lbpcascade_frontalface.xml")
    #result is a list of faces(rect parameters)
    faces = face_cascade.detectMultiScale(gray_image,scaleFactor =1.2,minNeighbours=5)

    #if no faces are detected then return original image, we return two None value because detect face fun returns two values
    if (len(faces) == 0):
      return None, None


    #under the assumption that there is only one face , then extract the face parameters
    (x, y, w, h) = faces[0]

    #return the face part of image
    return gray_image[y:y+w, x:x+h], faces[0]

#The goal of this fun is to prepare data, This function will read all person's images detect face from each image and will return two same size lists of faces and lables
def prepare_training_data(image_folder_path):
  #get all the directories inside the path
  dirs = os.listdir(image_folder_path)
  
  #define 2 lists to hold all subject faces and their corresponding lable
  faces = []
  lables = []

  #go throgh each directory and read the images
  for dir in dirs:
    #we know that our subject directories starts with 's', so we first ignore any none relevant directories
    if not dir.startwith("s"):
      continue
    
  #remove letter s from the directory name will give a lable and add the lable to the lable list
  lable =int(dir.replace("s",""))
  lables.append(lable)

  subject_dir_path = image_folder_path + "/" + dir #e.g. ~/s1

  #get all the images name in the subject directory
  subject_image_names = os.listdir(subject_dir_path)

  for image_name in subject_image_names:
    #first ignore system files like .ds_Store
    if image_name.startwith("."):
      continue
    
    #build image path
    image_path = subject_dir_path + "/" + image_name

    #read the image and display an image window to show the pic
    image = cv2.imread(image_path)
    cv2.imshow("Training on image ...", cv2.resize(image,(400,500)))
    cv2.waitkey(100)

    #pass the image to detect face fun
    face, rect = detect_face(image)

    #we should ignore faces that are not detected
    if face is not None:
      #add face to the faces list
      faces.append(face)
  
  cv2.destroyAllWindows()
  cv2.waitKey(1)
  cv2.destroyAllWindows()
  
print("End")    



  
