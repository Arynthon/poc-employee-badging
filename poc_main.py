import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import os

# Initialize database and logging file
LOG_FILE = "employee_logs.csv"
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["Name", "Status", "Timestamp"])
    df.to_csv(LOG_FILE, index=False)

# Load known employee faces
KNOWN_FACES_DIR = "known_faces"  # Folder with images of employees (named as their IDs or names)
known_faces = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(filepath)
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(filename)[0])

def log_event(name, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(LOG_FILE)
    df = pd.concat([df, pd.DataFrame([[name, status, timestamp]], columns=df.columns)])
    df.to_csv(LOG_FILE, index=False)
    print(f"{name} {status} at {timestamp}")

def process_frame(frame):
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB for face_recognition
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        df = pd.read_csv(LOG_FILE)
        last_entry = df[df["Name"] == name].iloc[-1] if not df[df["Name"] == name].empty else None

        if last_entry is None or last_entry["Status"] == "Signed Out":
            log_event(name, "Signed In")
        else:
            log_event(name, "Signed Out")

        # Draw a rectangle and label on the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        cv2.imshow("Employee Badging System", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

