import cv2, numpy as np, torch
from fer_lib import (EMOTIONS, FER_MEAN, FER_STD, IMG_SIZE, build_model,
                     get_device)
import json, os

with open(os.path.join("artifacts", "best_config.json")) as f:
    cfg = json.load(f)
device = get_device()
model = build_model(cfg["architecture"], dropout=cfg["dropout"]).to(device)
model.load_state_dict(torch.load(os.path.join("artifacts", "best_tuned.pt"),
                                 map_location=device))
model.eval()

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
print("Press q to quit")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60)):
        face = cv2.resize(gray[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        t = torch.from_numpy((face - FER_MEAN) / FER_STD)[None, None].to(device)
        with torch.no_grad():
            probs = torch.softmax(model(t), 1)[0]
        i = int(probs.argmax())
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
        cv2.putText(frame, f"{EMOTIONS[i]} {probs[i]:.0%}", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.imshow("FER demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release(); cv2.destroyAllWindows()