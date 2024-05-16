import pickle
import cv2
import mediapipe as mp
import numpy as np

# read model file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

# config mediapipe to detect landmarks of face
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face = mp_face.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)

while True:
    data_aux = []
    x_ = []
    y_ = []

    # convert bgr to rgb
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # find landmarks on face and draw
    results = face.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_CONTOURS,
                drawing_spec,
                drawing_spec)

        for face_landmarks in results.multi_face_landmarks:
            for i in range(len(face_landmarks.landmark)):
                x = face_landmarks.landmark[i].x
                y = face_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(face_landmarks.landmark)):
                x = face_landmarks.landmark[i].x
                y = face_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # take data to model for prediction
        prediction = model.predict([np.asarray(data_aux)])

        # map prediction to names
        predicted_name = prediction[0]

        # create rectangle on face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        # write name top of face
        cv2.putText(frame,
                    predicted_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break