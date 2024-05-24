import os
import cv2

number_of_faces = 2
dataset_size = 100

# make directory in parent/data path if not exist
path = os.path.dirname(os.getcwd())
DATA_DIR = path + '\\data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)
for i in range(number_of_faces):
    # make directory in ./data/{i} path if not exist
    name = input("write your name: ")
    if not os.path.exists(os.path.join(DATA_DIR, str(name))):
        os.makedirs(os.path.join(DATA_DIR, str(name)))

    print('Collecting data for class {}'.format(name))

    # opening camera and wait for q to start capturing
    while True:
        ret, frame = cap.read()
        cv2.putText(frame,
                    'press q to start',
                    (100, 50),
                    cv2.QT_FONT_NORMAL,
                    1.3,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # image counter
    counter = 0
    # start storing frames in directory
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(name), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
