import cv2
import numpy as np

# ============================================================
VIDEO_FILE = "fragment.mp4"
MODE       = "gray"
PAUSED     = False
# ============================================================

BUTTONS = [
    ("Original", "original"),
    ("Gray",     "gray"),
    ("Blur",     "blur"),
    ("HSV",      "hsv"),
    ("PAUSE",    "pause"),
]

BTN_H     = 40
BTN_PAD   = 10
BTN_RECTS = []


def draw_controls(panel_w):
    global BTN_RECTS
    BTN_RECTS = []
    panel = np.zeros((BTN_H + BTN_PAD * 2, panel_w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    n     = len(BUTTONS)
    btn_w = (panel_w - BTN_PAD * (n + 1)) // n
    x     = BTN_PAD

    for label, action in BUTTONS:
        y1, y2 = BTN_PAD, BTN_PAD + BTN_H
        x1, x2 = x, x + btn_w

        if action == MODE:
            bg = (0, 180, 0)
        elif action == "pause":
            bg = (0, 100, 200) if PAUSED else (0, 140, 255)
        else:
            bg = (80, 80, 80)

        cv2.rectangle(panel, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (200, 200, 200), 1)

        btn_label = ("RESUME" if PAUSED else "PAUSE") if action == "pause" else label
        (tw, th), _ = cv2.getTextSize(btn_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(panel, btn_label,
                    (x1 + (btn_w - tw) // 2, y1 + (BTN_H + th) // 2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        BTN_RECTS.append((x1, y1, x2, y2, action))
        x += btn_w + BTN_PAD

    return panel


def process_frame(frame):
    """Применяет текущий MODE к кадру, возвращает (оригинал_с_подписью, обработанный)."""
    font, scale, color, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    if MODE == "gray":
        proc  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proc  = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
        label = "Grayscale"
    elif MODE == "blur":
        proc  = cv2.GaussianBlur(frame, (15, 15), 0)
        label = "Gaussian Blur"
    elif MODE == "hsv":
        proc  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        label = "HSV"
    else:
        proc  = frame.copy()
        label = "Original"

    orig = frame.copy()
    cv2.putText(orig, "Original", (10, 30), font, scale, color, thick)
    cv2.putText(proc, label,      (10, 30), font, scale, color, thick)
    return orig, proc


def build_display(frame, frame_idx, total_frames, source_fps):
    orig, proc   = process_frame(frame)
    combined     = cv2.hconcat([orig, proc])
    info = f"Frame: {frame_idx}/{total_frames}  FPS: {source_fps:.1f}  Mode: {MODE}"
    cv2.putText(combined, info, (10, combined.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    panel = draw_controls(combined.shape[1])
    return cv2.vconcat([combined, panel]), combined.shape[0]


def on_mouse(event, x, y, flags, param):
    global MODE, PAUSED
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    video_h = param["video_h"]
    if y < video_h:
        return
    btn_y = y - video_h
    for x1, y1, x2, y2, action in BTN_RECTS:
        if x1 <= x <= x2 and y1 <= btn_y <= y2:
            if action == "pause":
                PAUSED = not PAUSED
            else:
                MODE = action
            param["needs_redraw"] = True
            break


cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print("Ошибка: не удалось открыть файл:", VIDEO_FILE)
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
source_fps   = cap.get(cv2.CAP_PROP_FPS)
delay        = max(1, int(1000 / source_fps))

WIN = "Video Processing"
cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

last_frame   = None
video_h      = None
mouse_state  = {"video_h": 0, "needs_redraw": False}

cv2.setMouseCallback(WIN, on_mouse, mouse_state)

print(f"Файл: {VIDEO_FILE}  |  Кадров: {total_frames}  |  FPS: {source_fps:.1f}")
print("Нажми [q] для выхода")
print("-" * 50)

while True:
    if not PAUSED:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        last_frame = frame
        frame_idx  = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        mouse_state["needs_redraw"] = False

    # Перерисовываем если на паузе нажали кнопку смены режима
    if PAUSED and mouse_state.get("needs_redraw"):
        mouse_state["needs_redraw"] = False

    if last_frame is not None:
        full, video_h = build_display(last_frame, frame_idx, total_frames, source_fps)
        mouse_state["video_h"] = video_h
        cv2.imshow(WIN, full)
        print(f"\rКадр: {frame_idx:5}/{total_frames}  |  Режим: {MODE}  |  {'⏸ ПАУЗА' if PAUSED else '▶ ИГРАЕТ'}  ", end="")

    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\nПрограмма завершена.")
