import cv2
import numpy as np
import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

# Попытка импорта YOLO
try:
    from ultralytics import YOLO

    print("⏳ Загрузка модели YOLO...")
    yolo_model = YOLO("yolov8n.pt")  # Легковесная модель YOLOv8
    print("✅ Модель YOLO успешно загружена.")
except ImportError:
    yolo_model = None
    print("⚠ ultralytics не установлен: pip install ultralytics")

# Попытка импорта аудио
try:
    from ffpyplayer.player import MediaPlayer

    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("⚠ ffpyplayer не установлен: pip install ffpyplayer")

# ============================================================
MODE = "original"
PAUSED = False
VIDEO_FILE = ""
audio_player = None

# Переменные для трекинга CSRT
tracker = None
tracker_bbox = None
init_tracker = False

# Переменные для оптического потока (Фарнебек)
prev_gray = None

# Переменные для сохранения видео
is_recording = False
video_writer = None
RECORD_OUT_PATH = "output_result.avi"
# ============================================================

BUTTONS = [
    ("Orig", "original"),
    ("YOLO", "yolo"),
    ("CSRT", "csrt"),
    ("Flow", "farneback"),
    ("REC", "record"),
    ("Help", "help"),
    ("PAUSE", "pause"),
    ("Open", "open"),
]

BTN_H = 40
BTN_PAD = 10
BTN_RECTS = []
SCRUB_H = 30
SCRUB_PAD = 10
SCRUB_RECT = [0, 0, 0, 0]
IS_DRAGGING = False


def pick_file():
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["osascript", "-e", "POSIX path of (choose file)"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                return path if path else None
        except Exception:
            pass
    elif sys.platform == "win32":
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$f=New-Object System.Windows.Forms.OpenFileDialog;"
            "$f.Filter='Video|*.mp4;*.avi;*.mkv;*.mov;*.wmv|All|*.*';"
            "if($f.ShowDialog() -eq 'OK'){$f.FileName}"
        )
        result = subprocess.run(["powershell", "-Command", ps],
                                capture_output=True, text=True)
        path = result.stdout.strip()
        return path if path else None
    print("Введи полный путь к видеофайлу:")
    path = input("> ").strip().strip('"').strip("'")
    return path if os.path.exists(path) else None


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть файл: {path}")
        return None, 0, 30, 33
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))
    return cap, total, fps, delay


def open_audio(path):
    global audio_player
    if not HAS_AUDIO:
        return
    if audio_player is not None:
        try:
            audio_player.close_player()
        except Exception:
            pass
    audio_player = MediaPlayer(path, ff_opts={"vn": True})


def show_help():
    root = tk.Tk()
    root.withdraw()
    help_text = (
        "ЛАБОРАТОРНАЯ РАБОТА №2\n"
        "Реализованные методы:\n\n"
        "1. YOLO (Детекция объектов): Нейронная сеть (You Only Look Once v8), способная находить "
        "объекты разных классов в реальном времени.\n\n"
        "2. CSRT (Трекинг объектов): Метод отслеживания (Channel and Spatial Reliability Tracker). "
        "При выборе этого режима откроется отдельное окно видео. Выделите объект мышью и нажмите SPACE или ENTER "
        "для начала отслеживания.\n\n"
        "3. Метод Фарнебека (Оптический поток): Плотный алгоритм оценки движения между кадрами. "
        "Результат представлен тепловой картой (где цвет означает направление, а яркость — скорость) "
        "и векторным полем (стрелками)."
    )
    messagebox.showinfo("Справка по методам", help_text)
    root.destroy()


def toggle_recording(width, height, fps):
    global is_recording, video_writer

    # Меняем расширение, если оно еще .avi
    out_path = RECORD_OUT_PATH.replace('.avi', '.mp4')

    if not is_recording:
        # Для Mac лучше всего подходит кодек mp4v (обязательно маленькими буквами)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        is_recording = True
        print(f"\n[REC] Запись начата: {out_path}")
    else:
        if video_writer:
            video_writer.release()
            video_writer = None
        is_recording = False
        print(f"\n[REC] Запись сохранена: {out_path}")

def reset_state():
    global tracker, tracker_bbox, prev_gray
    tracker = None
    tracker_bbox = None
    prev_gray = None


def process_frame(frame):
    global tracker, tracker_bbox, prev_gray
    font, scale, color, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    proc = frame.copy()
    label = "Original"

    # ==========================
    # 1. ДЕТЕКЦИЯ: YOLO
    # ==========================
    if MODE == "yolo" and yolo_model is not None:
        results = yolo_model(frame, verbose=False)
        proc = results[0].plot()
        label = "YOLO Detection"

    # ==========================
    # 2. ТРЕКИНГ: CSRT
    # ==========================
    elif MODE == "csrt":
        label = "CSRT Tracking"
        if tracker is not None:
            success, box = tracker.update(frame)
            if success:
                tracker_bbox = box
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(proc, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(proc, "Tracking", (x, y - 5), font, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(proc, "Lost Track", (50, 80), font, 1, (0, 0, 255), 2)

    # ==========================
    # 3. ОПТИЧЕСКИЙ ПОТОК: FARNEBACK
    # ==========================
    elif MODE == "farneback":
        label = "Farneback Flow"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            step = 16
            h, w = gray.shape
            y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)

            proc = cv2.addWeighted(frame, 0.5, heatmap, 0.8, 0)

            for (x1, y1), (x2, y2) in lines:
                if abs(x1 - x2) > 1 or abs(y1 - y2) > 1:
                    cv2.arrowedLine(proc, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.2)

        prev_gray = gray

    # ==========================

    orig = frame.copy()
    cv2.putText(orig, "Original", (10, 30), font, scale, color, thick)
    cv2.putText(proc, label, (10, 30), font, scale, color, thick)
    return orig, proc


def draw_controls(panel_w, video_name):
    global BTN_RECTS
    BTN_RECTS = []
    panel = np.zeros((BTN_H + BTN_PAD * 2, panel_w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    name = os.path.basename(video_name)
    cv2.putText(panel, name, (BTN_PAD, BTN_PAD + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    n = len(BUTTONS)
    btn_w = (panel_w - BTN_PAD * (n + 1)) // n
    x = BTN_PAD
    for label, action in BUTTONS:
        y1, y2 = BTN_PAD, BTN_PAD + BTN_H
        x1, x2 = x, x + btn_w

        if action == "open":
            bg = (60, 60, 140)
        elif action == "help":
            bg = (140, 100, 60)
        elif action == "record":
            bg = (0, 0, 200) if is_recording else (60, 60, 60)
        elif action == MODE:
            bg = (0, 180, 0)
        elif action == "pause":
            bg = (0, 100, 200) if PAUSED else (0, 140, 255)
        else:
            bg = (80, 80, 80)

        cv2.rectangle(panel, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (200, 200, 200), 1)

        btn_label = label
        if action == "pause": btn_label = "RESUME" if PAUSED else "PAUSE"
        if action == "record": btn_label = "STOP" if is_recording else "REC (Save)"

        (tw, th), _ = cv2.getTextSize(btn_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(panel, btn_label,
                    (x1 + (btn_w - tw) // 2, y1 + (BTN_H + th) // 2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        BTN_RECTS.append((x1, y1, x2, y2, action))
        x += btn_w + BTN_PAD
    return panel


def draw_scrubber(panel_w, frame_idx, total_frames, source_fps):
    global SCRUB_RECT
    panel = np.zeros((SCRUB_H + SCRUB_PAD * 2, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    bar_x1 = SCRUB_PAD
    bar_x2 = panel_w - SCRUB_PAD
    bar_y = (SCRUB_H + SCRUB_PAD * 2) // 2
    bar_h = 4
    cv2.rectangle(panel, (bar_x1, bar_y - bar_h // 2), (bar_x2, bar_y + bar_h // 2), (90, 90, 90), -1)
    progress = frame_idx / max(total_frames, 1)
    fill_x = int(bar_x1 + (bar_x2 - bar_x1) * progress)
    cv2.rectangle(panel, (bar_x1, bar_y - bar_h // 2), (fill_x, bar_y + bar_h // 2), (0, 180, 255), -1)
    handle_r = 8
    cv2.circle(panel, (fill_x, bar_y), handle_r, (0, 200, 255), -1)
    cv2.circle(panel, (fill_x, bar_y), handle_r, (255, 255, 255), 1)
    cur_sec = frame_idx / max(source_fps, 1)
    tot_sec = total_frames / max(source_fps, 1)
    t_str = f"{int(cur_sec) // 60:02}:{int(cur_sec) % 60:02} / {int(tot_sec) // 60:02}:{int(tot_sec) % 60:02}"

    status_str = f" | REC: {'ON' if is_recording else 'OFF'}"
    cv2.putText(panel, t_str + status_str, (bar_x1, SCRUB_PAD - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    SCRUB_RECT = [bar_x1, bar_y - handle_r - 4, bar_x2, bar_y + handle_r + 4]
    return panel


def build_display(frame, frame_idx, total_frames, source_fps, video_name):
    orig, proc = process_frame(frame)
    combined = cv2.hconcat([orig, proc])

    if is_recording and video_writer is not None:
        video_writer.write(combined)

    info = f"Frame: {frame_idx}/{total_frames}  FPS: {source_fps:.1f}  Mode: {MODE}"
    cv2.putText(combined, info, (10, combined.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    w = combined.shape[1]
    panel = draw_controls(w, video_name)
    scrub = draw_scrubber(w, frame_idx, total_frames, source_fps)
    full = cv2.vconcat([combined, panel, scrub])
    return full, combined.shape[0], combined.shape[0] + panel.shape[0]


def seek_to_x(mouse_x, panel_w, cap, total_frames, state):
    global audio_player
    bar_x1 = SCRUB_PAD
    bar_x2 = panel_w - SCRUB_PAD
    ratio = (mouse_x - bar_x1) / max(bar_x2 - bar_x1, 1)
    ratio = max(0.0, min(1.0, ratio))
    target = int(ratio * total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    if ret:
        state["seek_frame"] = frame
        state["seek_idx"] = target
        reset_state()
        if HAS_AUDIO and audio_player:
            seek_sec = target / max(state.get("source_fps", 30), 1)
            audio_player.seek(seek_sec, relative=False)


def on_mouse(event, x, y, flags, param):
    global MODE, PAUSED, IS_DRAGGING, audio_player, init_tracker
    cap = param["cap"]
    total_frames = param["total_frames"]
    panel_w = param["panel_w"]
    video_h = param["video_h"]
    btn_panel_h = param["btn_panel_h"]
    fps = param["source_fps"]
    scrub_y = video_h + btn_panel_h

    if event == cv2.EVENT_LBUTTONDOWN and video_h <= y < scrub_y:
        btn_y = y - video_h
        for x1, y1, x2, y2, action in BTN_RECTS:
            if x1 <= x <= x2 and y1 <= btn_y <= y2:
                if action == "pause":
                    PAUSED = not PAUSED
                    if HAS_AUDIO and audio_player:
                        audio_player.set_pause(PAUSED)
                elif action == "open":
                    param["open_file"] = True
                elif action == "help":
                    show_help()
                elif action == "record":
                    tw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
                    th = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    toggle_recording(tw, th, fps)
                else:
                    if action != MODE:
                        reset_state()
                        MODE = action
                        if MODE == "csrt":
                            init_tracker = True
                break

    if y >= scrub_y:
        local_y = y - scrub_y
        sx1, sy1, sx2, sy2 = SCRUB_RECT
        in_scrub = sx1 <= x <= sx2 and sy1 <= local_y <= sy2
        if event == cv2.EVENT_LBUTTONDOWN and in_scrub:
            IS_DRAGGING = True
            PAUSED = True
            if HAS_AUDIO and audio_player: audio_player.set_pause(True)
            seek_to_x(x, panel_w, cap, total_frames, param)
        elif event == cv2.EVENT_MOUSEMOVE and IS_DRAGGING:
            seek_to_x(x, panel_w, cap, total_frames, param)
        elif event == cv2.EVENT_LBUTTONUP and IS_DRAGGING:
            IS_DRAGGING = False
            PAUSED = False
            if HAS_AUDIO and audio_player: audio_player.set_pause(False)
            seek_to_x(x, panel_w, cap, total_frames, param)


# ── Запуск ──────────────────────────────────────────────────
print("Выбери видеофайл...")
chosen = pick_file()
if not chosen:
    print("Файл не выбран, выход.")
    exit(0)
VIDEO_FILE = chosen

cap, total_frames, source_fps, delay = open_video(VIDEO_FILE)
if cap is None:
    exit(1)

open_audio(VIDEO_FILE)

WIN = "Video Processing"
cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

frame_idx = 0
last_frame = None
video_h = 0
btn_panel_h = BTN_H + BTN_PAD * 2

mouse_state = {
    "video_h": 0, "btn_panel_h": btn_panel_h,
    "cap": cap, "total_frames": total_frames, "panel_w": 1,
    "source_fps": source_fps,
    "seek_frame": None, "seek_idx": None,
    "open_file": False,
}
cv2.setMouseCallback(WIN, on_mouse, mouse_state)

print("\nУПРАВЛЕНИЕ:")
print("- Кнопка [Help] выводит справку по методам.")
print("- Кнопка [REC] включает/выключает сохранение видео.")
print("- Для метода CSRT: Нажмите на кнопку [CSRT], выделите объект мышкой, нажмите ПРОБЕЛ или ENTER.")
print("- Нажми [q] для выхода.\n" + "-" * 50)

while True:
    if mouse_state["open_file"]:
        mouse_state["open_file"] = False
        new_path = pick_file()
        if new_path:
            cap.release()
            cap, total_frames, source_fps, delay = open_video(new_path)
            if cap is None: break
            open_audio(new_path)
            VIDEO_FILE = new_path
            frame_idx = 0
            last_frame = None
            PAUSED = False
            reset_state()
            mouse_state.update({"cap": cap, "total_frames": total_frames, "source_fps": source_fps, "seek_frame": None,
                                "seek_idx": None})

    if mouse_state["seek_frame"] is not None:
        last_frame = mouse_state["seek_frame"]
        frame_idx = mouse_state["seek_idx"]
        mouse_state["seek_frame"] = None
        mouse_state["seek_idx"] = None

    elif not PAUSED:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            reset_state()
            if HAS_AUDIO and audio_player:
                audio_player.seek(0, relative=False)
            continue
        last_frame = frame
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if last_frame is not None:
        # ---------------------------------------------------------
        # ФИКС ДЛЯ CSRT: Выбор зоны перенесен в главный цикл
        # ---------------------------------------------------------
        if init_tracker:
            init_tracker = False
            PAUSED = True
            if HAS_AUDIO and audio_player: audio_player.set_pause(True)

            print("\n[CSRT] В появившемся окне выделите объект. Нажмите SPACE или ENTER.")
            # Используем ОТДЕЛЬНОЕ окно, чтобы не ломать наш интерфейс с кнопками
            bbox = cv2.selectROI("Select Object CSRT", last_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Object CSRT")
            cv2.waitKey(1)  # Обязательно для macOS, чтобы окно корректно закрылось

            # На всякий случай переподключаем мышь к главному окну
            cv2.setMouseCallback(WIN, on_mouse, mouse_state)

            if bbox[2] > 0 and bbox[3] > 0:
                try:
                    if hasattr(cv2, 'TrackerCSRT_create'):
                        tracker = cv2.TrackerCSRT_create()
                    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                        tracker = cv2.legacy.TrackerCSRT_create()
                    else:
                        raise AttributeError
                    tracker.init(last_frame, bbox)
                except AttributeError:
                    print("⚠ Ошибка: Модуль CSRT не найден!")
                    tracker = None

            PAUSED = False
            if HAS_AUDIO and audio_player: audio_player.set_pause(False)
        # ---------------------------------------------------------

        full, video_h, _ = build_display(last_frame, frame_idx, total_frames, source_fps, VIDEO_FILE)
        mouse_state["video_h"] = video_h
        mouse_state["panel_w"] = full.shape[1]
        cv2.imshow(WIN, full)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break

if is_recording and video_writer:
    video_writer.release()
if HAS_AUDIO and audio_player:
    audio_player.close_player()
cap.release()
cv2.destroyAllWindows()
print("\nПрограмма завершена.")