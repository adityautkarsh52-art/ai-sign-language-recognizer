
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import model_from_json

# Load model
with open("model_class.json","r") as f:
    model = model_from_json(f.read())

model.load_weights("final_model.h5")

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    label = "No hand"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            h, w, _ = frame.shape
            xs, ys = [], []

            for lm in handLms.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            roi = frame[y1:y2, x1:x2]

            if roi.size != 0:
                img = cv2.resize(roi,(64,64))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = img/255.0
                img = img.reshape(1,64,64,1)

                pred = model.predict(img, verbose=0)
                label = labels[np.argmax(pred)]

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Prediction: {label}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("AI Sign Language Recognizer",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
