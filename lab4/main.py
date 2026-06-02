import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    is_new_cv2 = False
except AttributeError:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    is_new_cv2 = True

if is_new_cv2 and hasattr(cv2.aruco, 'ArucoDetector'):
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
else:
    detector = None

MODE = "original"
is_recording = False
writer = None
cap = None

BUTTONS = ["Load", "Orig", "ArUco", "Pose", "Save", "Help"]
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
                (text == "ArUco" and MODE == "aruco") or \
                (text == "Pose" and MODE == "pose"):
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
    global MODE, is_recording, writer, cap
    vh = param["vh"]
    if event == cv2.EVENT_LBUTTONDOWN and y >= vh:
        for x1, x2, text in BTN_RECTS:
            if x1 <= x <= x2:
                if text == "Load":
                    path = load_video()
                    if path:
                        if cap: cap.release()
                        cap = cv2.VideoCapture(path)
                elif text == "Orig":
                    MODE = "original"
                elif text == "ArUco":
                    MODE = "aruco"
                elif text == "Pose":
                    MODE = "pose"
                elif text == "Save":
                    if not is_recording and cap:
                        cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        writer = cv2.VideoWriter("output_lab4.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (cw, ch))
                        is_recording = True
                    else:
                        if writer: writer.release()
                        is_recording = False
                elif text == "Help":
                    root = tk.Tk()
                    root.withdraw()
                    messagebox.showinfo("Справка", "ArUco: Границы и ID.\nPose: Оси координат.")
                    root.destroy()
                break


path = load_video()
if not path: exit()
cap = cv2.VideoCapture(path)
cv2.namedWindow("Lab4", cv2.WINDOW_AUTOSIZE)

while True:
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        proc = frame.copy()
        if MODE in ["aruco", "pose"]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if detector is not None:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(proc, corners, ids)

                if MODE == "pose":
                    h, w = frame.shape[:2]
                    focal_length = w
                    camera_matrix = np.array([[focal_length, 0, w / 2],
                                              [0, focal_length, h / 2],
                                              [0, 0, 1]], dtype=float)
                    dist_coeffs = np.zeros((4, 1))
                    marker_length = 0.1

                    if hasattr(cv2.aruco, 'estimatePoseSingleMarkers'):
                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix,
                                                                              dist_coeffs)
                        for rvec, tvec in zip(rvecs, tvecs):
                            if hasattr(cv2, 'drawFrameAxes'):
                                cv2.drawFrameAxes(proc, camera_matrix, dist_coeffs, rvec, tvec, marker_length)
                            else:
                                cv2.aruco.drawAxis(proc, camera_matrix, dist_coeffs, rvec, tvec, marker_length)
                    else:
                        obj_points = np.array([[-marker_length / 2, marker_length / 2, 0],
                                               [marker_length / 2, marker_length / 2, 0],
                                               [marker_length / 2, -marker_length / 2, 0],
                                               [-marker_length / 2, -marker_length / 2, 0]], dtype=np.float32)
                        for corner in corners:
                            _, rvec, tvec = cv2.solvePnP(obj_points, corner[0], camera_matrix, dist_coeffs)
                            cv2.drawFrameAxes(proc, camera_matrix, dist_coeffs, rvec, tvec, marker_length)

        if is_recording and writer:
            writer.write(proc)

        panel = draw_buttons(proc.shape[1])
        full = cv2.vconcat([proc, panel])
        cv2.setMouseCallback("Lab4", on_mouse, {"vh": proc.shape[0]})
        cv2.imshow("Lab4", full)

    if cv2.waitKey(30) & 0xFF == ord('q'): break

if writer: writer.release()
if cap: cap.release()
cv2.destroyAllWindows()