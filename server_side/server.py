import json
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import socket
import struct
import threading

def handle_client(client_socket, face):
    received_data = b""
    payload_size = struct.calcsize("L")

    while True:
        try:
            # Ensure enough data is available to read the message size
            while len(received_data) < payload_size:
                data = client_socket.recv(4096)
                if not data:
                    # Client has disconnected
                    print("Client disconnected")
                    return
                received_data += data

            # Extract the packed message size
            packed_msg_size = received_data[:payload_size]
            received_data = received_data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            # Receive and assemble the frame data until the complete frame is received
            while len(received_data) < msg_size:
                data = client_socket.recv(4096)
                if not data:
                    # Client has disconnected
                    print("Client disconnected")
                    return
                received_data += data

            # Extract the frame data
            frame_data = received_data[:msg_size]
            received_data = received_data[msg_size:]

            # Deserialize the received frame
            frame = pickle.loads(frame_data)

            # Display the received frame
            # cv2.imshow('Client Video', received_frame)

            data_aux = []
            x_ = []
            y_ = []

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # find landmarks on face and draw
            results = face.process(frame_rgb)
            if results.multi_face_landmarks:
                # for face_landmarks in results.multi_face_landmarks:
                #     mp_drawing.draw_landmarks(
                #         frame,
                #         face_landmarks,
                #         mp_face.FACEMESH_CONTOURS,
                #         drawing_spec,
                #         drawing_spec)

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

                # Process the frame and send a response back to the client
                response = {"prediction": prediction[0], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                response_data = json.dumps(response).encode('utf-8')
                response_msg_size = struct.pack("L", len(response_data))
                client_socket.sendall(response_msg_size + response_data)

            # Press ‘q’ to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Client connection error: {e}")
            break

    client_socket.close()
    cv2.destroyAllWindows()

path = os.path.dirname(os.getcwd())

ip = 'localhost'
port = 8080

# read model file
model_dict = pickle.load(open(path + '\\model.p', 'rb'))
model = model_dict['model']

# cap = cv2.VideoCapture(0)

# config mediapipe to detect landmarks of face
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face = mp_face.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((ip, port))
server_socket.listen(5)

print("Server is listening on port 8080 ...")

while True:
    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address} has been established.")
    client_thread = threading.Thread(target=handle_client, args=(client_socket,face))
    client_thread.start()