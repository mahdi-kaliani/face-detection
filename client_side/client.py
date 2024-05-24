import json
import pickle
import socket
import struct
import cv2

ip = 'localhost'
port = 8080

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ip, port))

print(f"connect to {ip}:{port} ...")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Serialize the frame
    data = pickle.dumps(frame)
    message_size = struct.pack("L", len(data))

    # Send frame to server
    client_socket.sendall(message_size + data)

    # Receive response from server
    received_data = b""
    payload_size = struct.calcsize("L")

    while len(received_data) < payload_size:
        received_data += client_socket.recv(4096)

    packed_msg_size = received_data[:payload_size]
    received_data = received_data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    while len(received_data) < msg_size:
        received_data += client_socket.recv(4096)

    response_data = received_data[:msg_size]
    response = json.loads(response_data.decode('utf-8'))

    prediction = response.get('prediction')
    x1 = response.get('x1')
    y1 = response.get('y1')
    x2 = response.get('x2')
    y2 = response.get('y2')

    # create rectangle on face
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    # write name top of face
    cv2.putText(frame,
                prediction,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 0),
                3,
                cv2.LINE_AA)

    # Display the frame being sent (optional)
    cv2.imshow('Video', frame)

    # Press ‘q’ to quit
    if cv2.waitKey(1) == ord('q'):
        break

print(f"disconnect from {ip}:{port}")

cap.release()
cv2.destroyAllWindows()
client_socket.close()
