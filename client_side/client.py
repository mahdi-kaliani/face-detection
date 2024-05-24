import json
import pickle
import socket
import struct
import cv2
import threading

def recv_all(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError('Was expecting %d bytes but only received %d bytes before the socket closed' % (length, len(data)))
        data += more
    return data

def receive_response(client_socket):
    global prediction, x1, y1, x2, y2
    payload_size = struct.calcsize("L")

    while True:
        try:
            packed_msg_size = recv_all(client_socket, payload_size)
            msg_size = struct.unpack("L", packed_msg_size)[0]
            response_data = recv_all(client_socket, msg_size)
            response = json.loads(response_data.decode('utf-8'))

            prediction = response.get('prediction')
            x1 = response.get('x1')
            y1 = response.get('y1')
            x2 = response.get('x2')
            y2 = response.get('y2')

        except Exception as e:
            print(f"Error receiving data: {e}")
            break

ip = 'localhost'
port = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ip, port))

print(f"Connected to {ip}:{port} ...")

# Start a thread to receive responses from the server
response_thread = threading.Thread(target=receive_response, args=(client_socket,))
response_thread.daemon = True
response_thread.start()

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Serialize the frame
        data = pickle.dumps(frame)
        message_size = struct.pack("L", len(data))

        # Send frame to server
        client_socket.sendall(message_size + data)

        # Draw the rectangle and prediction text if they are available
        if 'prediction' in globals():
            # Create rectangle on face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            # Write name top of face
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

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print(f"Disconnected from {ip}:{port}")
    cap.release()
    cv2.destroyAllWindows()
    client_socket.close()
