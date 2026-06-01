import cv2
import numpy as np
import tkinter as tk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tkinter import filedialog, messagebox
import torch
from torchvision import models, transforms
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights).to(device).eval()
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
np.random.seed(42)
COLOR_MAP = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
COLOR_MAP[0] = [0, 0, 0]

yolo = YOLO("yolov8n.pt")

MODE = "original"
is_recording = False
writer = None
seen_ids = set()
cap = None

BUTTONS = ["Load", "Orig", "DeepLab", "Count", "Save", "Help"]
BTN_H = 40
BTN_RECTS = []


def load_video():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov")])
    root.destroy()
    return path


def draw_buttons(w):
    global BTN_RECTS
    BTN_RECTS = []
    panel = np.zeros((BTN_H, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    bw = w // len(BUTTONS)
    for i, text in enumerate(BUTTONS):
        x1, x2 = i * bw, (i + 1) * bw
        bg = (80, 80, 80)
        if text == "Save" and is_recording:
            bg = (0, 0, 200)
        elif (text == "Orig" and MODE == "original") or \
                (text == "DeepLab" and MODE == "deeplab") or \
                (text == "Count" and MODE == "count"):
            bg = (0, 180, 0)
        cv2.rectangle(panel, (x1, 0), (x2, BTN_H), bg, -1)
        cv2.rectangle(panel, (x1, 0), (x2, BTN_H), (200, 200, 200), 1)
        lbl = "Stop" if text == "Save" and is_recording else text
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(panel, lbl, (x1 + (bw - tw) // 2, (BTN_H + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        BTN_RECTS.append((x1, x2, text))
    return panel


def on_mouse(event, x, y, flags, param):
    global MODE, is_recording, writer, cap, seen_ids
    vh = param["vh"]
    if event == cv2.EVENT_LBUTTONDOWN and y >= vh:
        for x1, x2, text in BTN_RECTS:
            if x1 <= x <= x2:
                if text == "Load":
                    path = load_video()
                    if path:
                        if cap: cap.release()
                        cap = cv2.VideoCapture(path)
                        seen_ids = set()
                elif text == "Orig":
                    MODE = "original"
                elif text == "DeepLab":
                    MODE = "deeplab"
                elif text == "Count":
                    MODE = "count"
                elif text == "Save":
                    if not is_recording and cap:
                        cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        writer = cv2.VideoWriter("output_lab3.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (cw, ch))
                        is_recording = True
                    else:
                        if writer: writer.release()
                        is_recording = False
                elif text == "Help":
                    root = tk.Tk()
                    root.withdraw()
                    messagebox.showinfo("Справка", "DeepLab: Сегментация\nCount: Подсчет людей")
                    root.destroy()
                break


path = load_video()
if not path: exit()
cap = cv2.VideoCapture(path)
cv2.namedWindow("Lab3", cv2.WINDOW_AUTOSIZE)

while True:
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        proc = frame.copy()
        if MODE == "deeplab":
            h, w = frame.shape[:2]
            sf = 480 / max(h, w) if max(h, w) > 480 else 1.0
            sf_img = cv2.resize(frame, (int(w * sf), int(h * sf)))
            rgb = cv2.cvtColor(sf_img, cv2.COLOR_BGR2RGB)
            tensor = preprocess(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                out = deeplab(tensor)['out'][0]
            mask = out.argmax(0).byte().cpu().numpy()
            m_color = cv2.resize(COLOR_MAP[mask], (w, h), interpolation=cv2.INTER_NEAREST)
            proc = cv2.addWeighted(frame, 0.6, m_color, 0.6, 0)

        elif MODE == "count":
            res = yolo.track(frame, classes=[0], conf=0.15, persist=True, verbose=False)
            proc = res[0].plot()
            if res[0].boxes.id is not None:
                ids = res[0].boxes.id.cpu().numpy().astype(int)
                for i in ids: seen_ids.add(i)
            cv2.putText(proc, f"Persons: {len(seen_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if is_recording and writer:
            writer.write(proc)

        panel = draw_buttons(proc.shape[1])
        full = cv2.vconcat([proc, panel])
        cv2.setMouseCallback("Lab3", on_mouse, {"vh": proc.shape[0]})
        cv2.imshow("Lab3", full)

    if cv2.waitKey(30) & 0xFF == ord('q'): break

if writer: writer.release()
if cap: cap.release()
cv2.destroyAllWindows()