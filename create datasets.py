import os
import pickle
import mediapipe as mp
import cv2

DATA_DIR = './data'

# config mediapipe to detect landmarks of face
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)


data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    # opening each frames
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # find landmarks on face
        results = face.process(img_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for i in range(len(face_landmarks.landmark)):
                    # get x and y for dataset
                    x = face_landmarks.landmark[i].x
                    y = face_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(face_landmarks.landmark)):
                    x = face_landmarks.landmark[i].x
                    y = face_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # collect landmarks
            data.append(data_aux)
            labels.append(dir_)

# create data file
file = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, file)
file.close()
