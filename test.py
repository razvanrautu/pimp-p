import numpy as np
import cv2
import pickle
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

import threading
#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
pickle_in = open("model_trained.p", "rb")  ## rb = READ BYTE
model = pickle.load(pickle_in)

# Variabile pentru stocarea cadrului video și predicțiilor
current_frame = None
current_prediction = ""
current_probability = ""


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 45:
        return "Speed Limit 20 km/h"
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vechiles'
    elif classNo == 16:
        return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vechiles over 3.5 metric tons'

# Function to update the video frame
def update_video_frame():
    global current_frame
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_frame = frame  # Actualizăm cadrul video curent
        else:
            messagebox.showerror("Error", "Unable to read video frame!")

# Create the main GUI window
root = tk.Tk()
root.title("Traffic Sign Recognition")

# Create a label for displaying video
video_label = ttk.Label(root)
video_label.pack()

# Create a label for displaying the prediction result
result_label = ttk.Label(root, text="", font=("Helvetica", 16))
result_label.pack()

# Create a label for displaying the probability
probability_label = ttk.Label(root, text="", font=("Helvetica", 14))
probability_label.pack()

# Function to make predictions on the current frame
def predict_frame():
    global current_frame, current_prediction, current_probability
    while True:
        if current_frame is not None:
            frame = current_frame
            frame_copy = frame.copy()  # Facem o copie pentru a desena textul
            frame = cv2.resize(frame, (32, 32))
            frame = preprocessing(frame)
            frame = frame.reshape(1, 32, 32, 1)
            predictions = model.predict(frame)
            classIndex = np.argmax(predictions, axis=-1)
            probabilityValue = np.amax(predictions)
            if probabilityValue > threshold:
                current_prediction = getCalssName(classIndex)
            else:
                current_prediction = "Unknown"
            current_probability = str(round(probabilityValue * 100, 2)) + "%"
            # Desenăm rezultatele pe cadrul video
            cv2.putText(frame_copy, "CLASS: " + current_prediction, (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_copy, "PROBABILITY: " + current_probability, (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            # Actualizăm eticheta rezultatului în interfața grafică
            result_label.config(text=current_prediction)
            probability_label.config(text=current_probability)
            # Actualizăm eticheta video în interfața grafică
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_copy))
            video_label.config(image=photo)
            video_label.image = photo
        else:
            current_prediction = "Unknown"
            current_probability = ""
            result_label.config(text=current_prediction)
            probability_label.config(text=current_probability)


# Creăm firele de execuție pentru citirea și predicția cadrului video
video_thread = threading.Thread(target=update_video_frame)
prediction_thread = threading.Thread(target=predict_frame)

# Pornim firele de execuție
video_thread.start()
prediction_thread.start()


# Button to quit the application
quit_button = ttk.Button(root, text="Quit Application", command=root.destroy)
quit_button.pack()

root.mainloop()