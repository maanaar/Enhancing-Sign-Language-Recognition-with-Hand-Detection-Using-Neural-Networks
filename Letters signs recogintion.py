import cv2
import numpy as np
from keras.models import load_model
import time
# Define the size of the input images
size = 64

# Load the trained model
model = load_model('C:/Users/mosta/Downloads/Telegram Desktop/fvf/lastarabi.h5')

# Define a dictionary to map class labels to class names
class_names = {' ': 0,
 'أ': 1,
 'ئ': 2,
 'ال': 3,
 'ب': 4,
 'ة': 5,
 'ت': 6,
 'ث': 7,
 'ج': 8,
 'ح': 9,
 'خ': 10,
 'د': 11,
 'ذ': 12,
 'ر': 13,
 'ز': 14,
 'س': 15,
 'ش': 16,
 'ص': 17,
 'ض': 18,
 'ط': 19,
 'ظ': 20,
 'ع': 21,
 'غ': 22,
 'ف': 23,
 'ق': 24,
 'ك': 25,
 'ل': 26,
 'لا': 27,
 'م': 28,
 'ن': 29,
 'ه': 30,
 'و': 31}
#,'ي': 32

# Initialize the video capture object
def cap():
    cap = cv2.VideoCapture(0)
    word = ''
    threshold = 0.97  # Set the threshold to 0.6
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        cv2.rectangle(frame, (400,400), (100,100), (0,255,0), 2)
        crop_img = frame[100:400, 100:400]
        # Convert the frame to grayscale
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # Resize the image to the input size of the model
        resized = cv2.resize(gray, (size, size))
        # Preprocess the image for input to the model
        preprocessed = np.expand_dims(resized, axis=-1)
        preprocessed = preprocessed.astype('float32') / 255.0
        # Make a prediction using the model
        prediction = model.predict(np.array([preprocessed]))
        # Get the predicted class label and class name
        predicted_label = np.argmax(prediction)
        predicted_prob = np.max(prediction)
        for key,value in class_names.items():
            if predicted_label == value:
                predicted_class=key
        # Check if the predicted probability is above the threshold
        if predicted_prob > threshold:
            word += predicted_class
        sentence_label.config(text=predicted_class)
        sentence_label2.config(text=word)
        root.update()
        # Draw the predicted class label on the frame
        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Display the frame
        cv2.imshow('frame', frame)
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
import cv2
import os
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
import mediapipe as mp
import numpy as np
from tensorflow import keras
import tkinter as tk


# Create the GUI
root = tk.Tk()
root.geometry("600x600")
root.title('Sign Language Detection')
# Create a label and text box to display the sentence output
output_label = tk.Label(root, text='Sentence Output:')
output_label.pack()

sentence_label = tk.Label(root, font=("Arial", 18), text='Sentence predict:')
sentence_label.pack(pady=40)

sentence_label2 = tk.Label(root, font=("Arial", 18), text='Sentence:')
sentence_label2.pack(pady=40)

# Create a button to start the sign language detection
start_button = tk.Button(root, text='Start', command=cap)
start_button.pack()

root.update()
root.mainloop()