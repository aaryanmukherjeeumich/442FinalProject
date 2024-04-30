import pickle
import time
import autocorrect
#pip install swig==3.0.6
#pip install jamspell   
# import jamspell
from textblob import TextBlob
import argparse
import textwrap

import cv2
import mediapipe as mp
import numpy as np


class HandData:
    def __init__(self, model):
        self.x_coors = []
        self.y_coors = []
        self.data = []
        self.last_change_time = time.time()
        self.current_char = None
        self.model = model

    def reset(self):
        self.x_coors = []
        self.y_coors = []
        self.data = []

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("-a", "--autocorrect", help = "Options: 1, 0", required = False, default = 1)
    argument = parser.parse_args()

    return int(argument.autocorrect)
    

def classify_user(auto_corr):

    # print("auto_corr: ", auto_corr)

    model_dict_right = pickle.load(open('./model_Right.p', 'rb'))
    model_right = model_dict_right['model']

    model_dict_left = pickle.load(open('./model_Left.p', 'rb'))
    model_left = model_dict_left['model']

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

    # last_change_time = time.time()
    start_time = time.time()
    # current_char = None
    word = ""

    # spell = autocorrect.Speller()
    # spell = jamspell.TSpellCorrector()
    # spell.LoadLangModel('en.bin')

    handDataDict = {
        "Left": HandData(model_left),
        "Right": HandData(model_right)
    }

    last_hand_time = time.time()
    _ , start_frame = cap.read()

    while True:

        handDataDict["Left"].reset()
        handDataDict["Right"].reset()


        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            last_hand_time = time.time()

            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmark, hand_type_data in zip(results.multi_hand_landmarks, results.multi_handedness):

                handType = hand_type_data.classification[0].label
                data = []
                for i in range(len(hand_landmark.landmark)):
                    x = hand_landmark.landmark[i].x
                    y = hand_landmark.landmark[i].y
                    handDataDict[handType].x_coors.append(x)
                    handDataDict[handType].y_coors.append(y)


                for i in range(len(hand_landmark.landmark)):
                    x = hand_landmark.landmark[i].x
                    y = hand_landmark.landmark[i].y

                    data.append(x - min(handDataDict[handType].x_coors))
                    data.append(y - min(handDataDict[handType].y_coors))

                handDataDict[handType].data = np.asarray(data)

            for handType in handDataDict:

                x_coors = handDataDict[handType].x_coors
                y_coors = handDataDict[handType].y_coors
                data = handDataDict[handType].data

                if len(x_coors) > 0 and len(y_coors) > 0 :

                    x1 = int(min(x_coors) * W) - 10
                    y1 = int(min(y_coors) * H) - 10

                    x2 = int(max(x_coors) * W) - 10
                    y2 = int(max(y_coors) * H) - 10

                    data = np.asarray(data).reshape(1, -1)

                    prediction = handDataDict[handType].model.predict(data)

                    if type(prediction[0]) is not np.int64 and len(prediction[0]) > 4:
                        # print("option 1")
                        max_val = np.argmax(prediction[0])
                        predicted_character = labels_dict[int(max_val)]

                    else:
                        # print("option 2")
                        predicted_character = labels_dict[int(prediction[0])]

                    # print("prediction: ", prediction)
                    # print("prediction[0]:", prediction[0])

                    if predicted_character != handDataDict[handType].current_char:
                        handDataDict[handType].current_char = predicted_character
                        handDataDict[handType].last_change_time = time.time()  # Update last change time
                    else:
                        if time.time() - handDataDict[handType].last_change_time >= 2:
                            word += handDataDict[handType].current_char
                            handDataDict[handType].last_change_time = time.time()

                    # if word hasnt changed in 10 seconds add a space

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
        else:  # No hand detected
            if time.time() - last_hand_time >= 1:
                handDataDict["Right"].last_change_time = time.time()
                handDataDict["Left"].last_change_time = time.time()

            if time.time() - last_hand_time >= 5:  # Check time since last hand detection
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

        H, W, _ = frame.shape

        wrapped_output_text = textwrap.wrap('Output : ' + (word.replace(" ", "-")), width=40)

        # wrapped_output_text = textwrap.wrap('Output: " + "the-quick-brown-fox-jumps-over-a-lazy-dog', width=50)
        gap = 0
        for i, line in enumerate(wrapped_output_text):
            gap = i
            cv2.putText(frame, line, (100, 50 + 50 * i), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 255, 100), 3,
                        cv2.LINE_AA)

        # cv2.putText(frame, 'Output : ' + (word.replace(" ", "-")), (100, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 255, 100), 3,
        #             cv2.LINE_AA)

        if auto_corr:
            corrected_text = "Autocorrect: " + (str(TextBlob(word.lower()).correct()))
            wrapped_corrected_text = textwrap.wrap(corrected_text, width= 40)
            # wrapped_corrected_text = textwrap.wrap("the quick brown fox jumps over a lazy dog", width=W * 9//10)

            for i, line in enumerate(wrapped_corrected_text):

                cv2.putText(frame, line, (100, 100 + 50 * (i+gap)), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 255, 100), 3,
                    cv2.LINE_AA)

        cv2.putText(start_frame, "Press 'q' to quit", (int(W/2 - 250), int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(start_frame, "Remove hands from view for 10 sec. to add a space", (int(W/2 - 250), int(H/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(start_frame, "Beginning in 5 seconds", (int(W/2 - 250), int(H/2)+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if time.time() - start_time > 5:
            start_frame = frame
    
        cv2.imshow('frame',start_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("flag 1")
    auto_corr = parse_command_line_arguments()
    print("flag 2")

    classify_user(auto_corr)