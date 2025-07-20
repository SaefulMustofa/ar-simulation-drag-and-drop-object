import cv2
import mediapipe as mp
import math
import time

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Kamera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Gambar hologram
holo_img = cv2.imread("object1.png", cv2.IMREAD_UNCHANGED)  # Baca dengan alpha channel
holo_size = 100  # Ukuran gambar

# Fungsi untuk overlay gambar dengan alpha (transparan)
def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]

    if overlay.shape[2] < 4:
        return background  # Bukan gambar transparan

    # Area target
    roi = background[y:y+h, x:x+w]
    
    # Pisah alpha channel
    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0

    # Blend
    background[y:y+h, x:x+w] = (1.0 - mask) * roi + mask * overlay_img
    return background

# Posisi gambar
holo_pos = [300, 300]
dragging = False

# FPS
pTime = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for i, handLms in enumerate(result.multi_hand_landmarks):
            hand_label_raw = result.multi_handedness[i].classification[0].label
            hand_label = "Kanan" if hand_label_raw == "Right" else "Kiri"

            if hand_label == "Kanan":
                h, w, _ = frame.shape
                x1 = int(handLms.landmark[8].x * w)
                y1 = int(handLms.landmark[8].y * h)
                x2 = int(handLms.landmark[4].x * w)
                y2 = int(handLms.landmark[4].y * h)

                cv2.putText(frame, hand_label, (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                distance = math.hypot(x2 - x1, y2 - y1)

                if distance < 40:
                    dragging = True
                    # Tengah gambar = jari telunjuk
                    holo_pos = [x1 - holo_size // 2, y1 - holo_size // 2]
                    cv2.putText(frame, "MENARIK OBJEK", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 4)
                    cv2.putText(frame, "MENARIK OBJEK", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                else:
                    dragging = False

                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Tampilkan gambar hologram
    if holo_img is not None:
        frame = overlay_transparent(frame, cv2.resize(holo_img, (holo_size, holo_size)),
                                    holo_pos[0], holo_pos[1])

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Drag Gambar Hologram", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()