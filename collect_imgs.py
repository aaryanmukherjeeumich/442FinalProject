import os
import time
import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 1000

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

cap = cv2.VideoCapture(0)
for j in labels_dict:
    for hand_type in ["Right", "Left"]:

        if not os.path.exists(os.path.join(DATA_DIR, hand_type, str(j))):
            os.makedirs(os.path.join(DATA_DIR, hand_type, str(j)))

        print('Collecting data for class {}'.format(j))
        done = False
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape

            minY, maxY = H // 8, H - H // 8
            minX, maxX = W // 3, W - W // 3

            cv2.rectangle(frame, (minX,minY), (maxX,maxY), (0, 0, 0), 4)
            cv2.putText(frame, f'Ready? Press "Q" to begin! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

            cv2.putText(frame, f'Collecting data for character {labels_dict[j]} on {hand_type} hand', (50, H - H // 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1,
                        cv2.LINE_AA)

            #Collecting data for character {labels_dict[j]} on {hand_type} hand.
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape

            minY, maxY = H // 8, H - H // 8
            minX, maxX = W // 3, W - W // 3

            cv2.rectangle(frame, (minX,minY), (maxX,maxY), (0, 0, 0), 4)
            cv2.putText(frame, f'Character: {labels_dict[j]}, hand: {hand_type}, count: {counter}/{dataset_size}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, hand_type, str(j), '{}.jpg'.format(counter)), frame[minY:maxY, minX:maxX])
            time.sleep(0.01)
            counter += 1

cap.release()
cv2.destroyAllWindows()