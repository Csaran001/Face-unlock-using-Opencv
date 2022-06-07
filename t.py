import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import time
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

app = tk.Tk()
app.title('Login System')
app.geometry('500x350')
app.configure(bg="light green")



def openNewWindow():
    
    # Toplevel
    newWindow = Toplevel(app)
    newWindow.title("New Window")

    # sets the geometry of toplevel
    newWindow.geometry("200x200")

    # A Label widget to show in toplevel
    Label(newWindow,
        text ="Welcome").pack()


#face grayscale image samples
def save_samples():
    haar_file = 'haarcascade_frontalface_default.xml'

    # All the faces data will be
    #  present this folder

    datasets = 'C:\\Users\\ELCOT\\Desktop\\tk website\\'

    # These are sub data sets of folder,
    # for my faces I've used my name you can
    # change the label here

    sub_data = 'face_data'

    path = os.path.join(datasets, sub_data)

    if not os.path.isdir(path):
        os.mkdir(path)

    # the size of images

    (width, height) = (130, 100)

    # '0' is used for my webcam,
    # if you've any other camera
    #  attached use '1' like this

    face_cascade = cv2.CascadeClassifier(haar_file)

    webcam = cv2.VideoCapture(0)

    # The program loops until it has 30 images of the face.

    count = 1
    while count <= 100:

        (_, im) = webcam.read()

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face = gray[y:y + h, x:x + w]

            face_resize = cv2.resize(face, (width, height))

            cv2.imwrite('% s/% s.png' % (path, count), face_resize)

        count += 1

        cv2.imshow('OpenCV', im)

        key = cv2.waitKey(10)

        if key == 27:
            break



#training data
def face_id ():
    data_path = 'C:\\Users\\ELCOT\\Desktop\\tk website\\face_data\\'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    training_data, labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        training_data.append(np.asarray(images, dtype=np.uint8))
        labels.append(i)

    labels = np.asarray(labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(training_data), np.asarray(labels))
    print("Model training complete!!")

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_detector(img, size=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        if faces is ():
            return img, []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi

    cap = cv2.VideoCapture(0)


    s=0
    while s<= 1000:
        ret, frame = cap.read()
        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                display_string = str(confidence) + '% confidence it is user'
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

            if confidence > 70:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face cropper', image)
                print("welcome")
                if s>2:
                     openNewWindow()
                     print("welcome")
                     break
                s += 1

            else:
                cv2.putText(image, "locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face cropper', image)


        except:
            cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face cropper', image)
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


pic = Image.open("face.jpg")#(for image in main(app) page)
picture = ImageTk.PhotoImage(pic)

image =tk.Button(app, width=200, height=200, image=picture)
image.place(relx=0.5, rely=0.35, anchor=CENTER)

login = tk.Button(app, text='log In', width=20, height=3,activebackground="dark grey", activeforeground="red", relief=GROOVE, command = face_id )
login.place(relx=0.3, rely=0.75, anchor=CENTER)

register = tk.Button(app, text='Register', width=20, height=3,activebackground="dark grey", activeforeground="red", relief=GROOVE, command = save_samples)
register.place(relx=0.7, rely=0.75, anchor=CENTER)

stop = tk.Button(app, text='Exit', width=20, command=app.destroy,bg="red", activebackground="red", relief=GROOVE)
stop.place(relx=0.3, rely=1, anchor=SE)



app.mainloop()
