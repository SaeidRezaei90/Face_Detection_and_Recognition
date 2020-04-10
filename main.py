# import opencv module
import cv2
# import os module for reading data directories and paths
import os
# import numpy to convert python lists to numpy arrays as it is needed by openCV file recognation
import numpy as np

# there is no label 0 in pur training data so subject name for index 0 is empt
subjects = ["", "Ronaldo", "Messi", "Robben"]


# Function to detect face using openCV
def detect_face(img):
    # Convert the image to a gray image as openCV face detector expects gray image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load opencv face detector usin LBP which is fast
    face_cascade = cv2.CascadeClassifier(
        "opencv-files/lbpcascade_frontalface.xml")
    # result is a list of faces(rect parameters)
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.2, minNeighbours=5)

    # if no faces are detected then return original image, we return two None value because detect face fun returns two values
    if (len(faces) == 0):
        return None, None

    # under the assumption that there is only one face , then extract the face parameters
    (x, y, w, h) = faces[0]

    # return the face part of image
    return gray_image[y:y + w, x:x + h], faces[0]


# # # # # # # # # # # # # # # # # #  END OF THE FACE DETECT FUNCTION # # # # # # # # # # # # # # # # # # # # # # # 


# The goal of this fun is to prepare data, This function will read all person's images detect face from each image and will return two same size lists of faces and labels
def prepare_training_data(image_folder_path):
    # get all the directories inside the path
    dirs = os.listdir(image_folder_path)

    # define 2 lists to hold all subject faces and their corresponding label
    faces = []
    labels = []

    # go throgh each directory and read the images
    for dir in dirs:
        # we know that our subject directories starts with 's', so we first ignore any none relevant directories
        if not dir.startswith("s"):
            continue

        # remove letter s from the directory name will give a label
        label = int(dir.replace("s", ""))
        subject_dir_path = image_folder_path + "/" + dir  # e.g. ~/s1

        # get all the images name in the subject directory
        subject_image_names = os.listdir(subject_dir_path)

        for image_name in subject_image_names:
            # first ignore system files like .ds_Store
            if image_name.startswith("."):
                continue
            
            # build image path
            image_path = subject_dir_path + "/" + image_name
            
            print(image_path)
            # read the image and display an image window to show the pic
            image = cv2.imread(image_path)
            

            image = cv2.resize(image, (400, 500))
            cv2.imshow('Training on image ...', image)
            cv2.waitkey(100)

            # pass the image to detect face fun
            face, rect = detect_face(image)

            # we should ignore faces that are not detected
            if face is not None:
                # add face to the faces list and label to the label list
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


# # # # # # # # # # # # # # # # # #  END OF THE PREPARE TRAINING DATA FUNCTION # # # # # # # # # # # # # # # # # # # # # # # 

print("Priparing data ....")
# training_path = os.path.join('C:/' + 'Users')
faces, labels = prepare_training_data('trainingdata')
print("data prepared")

# print total faces and numbers
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# creat LBPH face recognizer
face_recognizer = cv2.creatLBPHFaceRecognizer()
face_recognizer.train(faces, np.array(labels))


# define 2 function to draw rectangle on image according to given (x,y) coordinates and draw text near that
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# define a function to recognize the person in image and draw a rectangle in the image and write the name of the person
def predict(test_img):
    # make a cope of the image as we dont want to change the original pic1
    img = test_img.copy()

    # detect face from the cope
    face , rect = detect_face(img)

    # predict the image using face recognizer
    label, confidence = face_recognizer.predict(face)
    print(confidence)

    # Get name of respective label
    label_text = subjects[label]

    if confidence < 30:
      label_text = "not able to recognize"
    
    # draw a rectangle a name of the pic 
    draw_rectangle(face,rect)
    draw_text(img, label_text, rect[0], rect[1]-5)

    return img

print("predict images ....")

# lacal test images
test_image1 = cv2.imread(os.path.join("TestImages",'messi.jpg'))
test_image2 = cv2.imread(os.path.join("TestImages",'Ronaldo.jpg'))

# perform a prediction
predicted_img1 = predict(test_image1)
predicted_img2 = predict(test_image2)

print("prediction completed")

# display both images
cv2.imshow('image1', cv2.resize(predicted_img1, (400,500)))
cv2.imshow('image2', cv2.resize(predicted_img2, (400,500)))

cv2.waitKey(0)
cv2.destroyAllWindows

cv2.waitKey(1)
cv2.destroyAllWindows
