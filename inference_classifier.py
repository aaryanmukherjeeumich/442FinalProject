import pickle
import time
import autocorrect
#pip install swig==3.0.6
#pip install jamspell   
# import jamspell
from textblob import TextBlob



import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

last_change_time = time.time()
start_time = time.time()
current_char = None
word = ""

# spell = autocorrect.Speller()
# spell = jamspell.TSpellCorrector()
# spell.LoadLangModel('en.bin')
last_hand_time = time.time()
_ , start_frame = cap.read()

while True:

    # these should all be 2d arrays
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:

            sub_data_aux = []
            sub_x = []
            sub_y = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                sub_x.append(x)
                sub_y.append(y)

            x_.append(sub_x)
            y_.append(sub_y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                sub_data_aux.append(x - min(sub_x))
                sub_data_aux.append(y - min(sub_y))

            data_aux.append(np.asarray(sub_data_aux))

        for i in range(len(data_aux)):

            x_coors = x_[i]
            y_coors = y_[i]
            data = data_aux[i]


            x1 = int(min(x_coors) * W) - 10
            y1 = int(min(y_coors) * H) - 10

            x2 = int(max(x_coors) * W) - 10
            y2 = int(max(y_coors) * H) - 10

            prediction = model.predict([np.asarray(data)])

            predicted_character = labels_dict[int(prediction[0])]
            predicted_character = labels_dict[int(prediction[0])]

        if predicted_character != current_char:
            current_char = predicted_character
            last_change_time = time.time()  # Update last change time
        else:
            if time.time() - last_change_time >= 3:
                word += current_char
                last_change_time = time.time()
        

        


        # if word hasnt changed in 10 seconds add a space




        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    else:  # No hand detected
        if time.time() - last_hand_time >= 10:  # Check time since last hand detection
            word += ' '  # Add a space
            last_hand_time = time.time()  # Update last time hand was detected
        # if time.time() - start_time < 10:
        #     cv2.putText(start_frame, "Press 'q' to quit", (int(W/2 - 250), int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #     cv2.putText(start_frame, "Remove hands from view to add a space", (int(W/2 - 250), int(H/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # else: 
        #     _ , start_frame = cap.read()
        


    # if time.time() - start_time < 10:
    #     cv2.putText(frame, "Press 'q' to quit", (int(W/2 - 250), int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     cv2.putText(frame, "Remove hands from view to add a space", (int(W/2 - 250), int(H/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
    cv2.putText(frame, 'Autocorrection : ' +  (str(TextBlob(word.lower()).correct())), (100, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 255, 100), 3,
                    cv2.LINE_AA)
    cv2.putText(frame, 'Output : ' + (word.replace(" ", "-")), (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 255, 100), 3,
                    cv2.LINE_AA)
    
    cv2.putText(start_frame, "Press 'q' to quit", (int(W/2 - 250), int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(start_frame, "Remove hands from view for 10 sec. to add a space", (int(W/2 - 250), int(H/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(start_frame, "Begining in 10 seconds", (int(W/2 - 250), int(H/2)+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    if time.time() - start_time > 10:
        start_frame = frame
    
    cv2.imshow('frame',start_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()