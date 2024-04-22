import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
i = 1
data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    print(dir_)
    # Sometimes the lines below need commented in/out try each just in case
    # if os.path.join(DATA_DIR, dir_) == "./data/.DS_Store":
    #     os.remove("./data/.DS_Store")
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # print(os.path.join(DATA_DIR, dir_, img_path))
        # print(img_path)
        if img_path == ".DS_Store":
            os.remove(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            
            for hand_landmarks in results.multi_hand_landmarks:
                # print((hand_landmarks.landmark))
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)
    

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
