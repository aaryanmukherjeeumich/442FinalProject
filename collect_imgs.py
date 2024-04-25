import os
import time
import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))
    done = False
    while True:
        ret, frame = cap.read()
        H, W, _ = frame.shape

        minY, maxY = H // 8, H - H // 8
        minX, maxX = W // 3, W - W // 3

        cv2.rectangle(frame, (minX,minY), (maxX,maxY), (0, 0, 0), 4)
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        H, W, _ = frame.shape

        minY, maxY = H // 8, H - H // 8
        minX, maxX = W // 3, W - W // 3

        cv2.rectangle(frame, (minX,minY), (maxX,maxY), (0, 0, 0), 4)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame[minY:maxY, minX:maxX])
        time.sleep(0.1)
        counter += 1

cap.release()
cv2.destroyAllWindows()