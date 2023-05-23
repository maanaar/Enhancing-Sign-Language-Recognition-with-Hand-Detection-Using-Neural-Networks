import cv2
import os
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
import mediapipe as mp
import numpy as np
from tensorflow import keras
import tkinter as tk
from tkinter import ttk
import itertools

actions = np.array(["صباحا","موجود","مباشر","الساعه","كام","مصر","السبت","فلوس","مقطع","تذكره","قطار"])
label_map = {label:num for num, label in enumerate(actions)}
# Load the LSTM model
model = keras.models.load_model('C:/Users/mosta/Downloads/ActionDetectionforSignLanguage-main/cam/1 test/927272%.h5')
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def mediapipe_draw(image,results):
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])





def cap(): 
            sequence = []
            threshold = 0.6
            sentence = []
            cap =cv2.VideoCapture(0)

            # Set mediapipe model 
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                 for i in itertools.count():

                    # Read feed
                    ret, frame = cap.read()
                    # Make detections
                    if not ret:
                         break
                    image, results = mediapipe_detection(frame, holistic)
                    print(results)
                    
                    # Draw landmarks
                    mediapipe_draw(image, results)
                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-50:]
                    
                    if len(sequence) == 50:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        print(actions[np.argmax(res)])
                        
                        
                    #3. Viz logic
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5: 
                            sentence = sentence[-5:]



                    sentence_str = ' '.join(sentence)
                    sentence_label.config(text=sentence_str)
                    root.update()
                    # Viz probabilities
                    cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()






# Create the GUI
root = tk.Tk()
root.geometry("600x600")
root.title('Sign Language Detection')
# Create a label and text box to display the sentence output
output_label = tk.Label(root, text='Sentence Output:')
output_label.pack()

sentence_label = tk.Label(root, font=("Arial", 18))
sentence_label.pack(pady=40)

# Create a button to start the sign language detection
start_button = tk.Button(root, text='Start', command=cap)
start_button.pack()

root.update()
root.mainloop()