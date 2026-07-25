import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import time
import os

BG_COLOR = "#1e1e2e"
FG_COLOR = "#cdd6f4"
ACCENT = "#89b4fa"
CARD_BG = "#313244"
BTN_BG = "#45475a"

class PotholeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pothole Detection System")
        self.root.geometry("1280x800")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(960, 600)

        self.model = YOLO("best.pt")
        self.model.fuse()

        self.cap = None
        self.is_playing = False
        self.video_path = None
        self.after_id = None

        self.conf_threshold = tk.DoubleVar(value=0.35)
        self.frame_count = 0
        self.detection_count = 0
        self.current_fps = 0.0
        self.last_time = time.time()
        self.frame_skip = 2
        self.processed_count = 0

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background=BG_COLOR)
        self.style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, font=("Segoe UI", 10))
        self.style.configure("TLabelframe", background=BG_COLOR, foreground=FG_COLOR, font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabelframe.Label", background=BG_COLOR, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        self.style.configure("TButton", background=BTN_BG, foreground=FG_COLOR, font=("Segoe UI", 10), borderwidth=0, focuscolor="none")
        self.style.map("TButton", background=[("active", ACCENT)])
        self.style.configure("TScale", background=BG_COLOR, troughcolor=CARD_BG, slidercolor=ACCENT)

        self.setup_ui()
        self.update_stats()

    def setup_ui(self):
        header = tk.Frame(self.root, bg=BG_COLOR, height=50)
        header.pack(fill=tk.X, padx=15, pady=(10, 0))

        tk.Label(header, text="Pothole Detection System", font=("Segoe UI", 16, "bold"),
                 fg=ACCENT, bg=BG_COLOR).pack(side=tk.LEFT)

        main = tk.Frame(self.root, bg=BG_COLOR)
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.video_container = tk.LabelFrame(main, text="Video Feed", bg=CARD_BG, fg=FG_COLOR,
                                              font=("Segoe UI", 10, "bold"), relief=tk.FLAT, bd=2)
        self.video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.video_container, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        right = tk.Frame(main, bg=BG_COLOR, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        right.pack_propagate(False)

        self.control_card = tk.LabelFrame(right, text="Controls", bg=CARD_BG, fg=FG_COLOR,
                                           font=("Segoe UI", 10, "bold"), relief=tk.FLAT, bd=2)
        self.control_card.pack(fill=tk.X, pady=(0, 10))

        self.open_btn = tk.Button(self.control_card, text="Open Video", command=self.open_video,
                                  bg=BTN_BG, fg=FG_COLOR, font=("Segoe UI", 10), relief=tk.FLAT,
                                  activebackground=ACCENT, activeforeground=BG_COLOR, cursor="hand2")
        self.open_btn.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.play_btn = tk.Button(self.control_card, text="Play", command=self.toggle_play,
                                  bg=BTN_BG, fg=FG_COLOR, font=("Segoe UI", 10), relief=tk.FLAT,
                                  activebackground=ACCENT, activeforeground=BG_COLOR, cursor="hand2", state=tk.DISABLED)
        self.play_btn.pack(fill=tk.X, padx=10, pady=5)

        self.stop_btn = tk.Button(self.control_card, text="Stop", command=self.stop,
                                  bg=BTN_BG, fg=FG_COLOR, font=("Segoe UI", 10), relief=tk.FLAT,
                                  activebackground=ACCENT, activeforeground=BG_COLOR, cursor="hand2", state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.params_card = tk.LabelFrame(right, text="Parameters", bg=CARD_BG, fg=FG_COLOR,
                                          font=("Segoe UI", 10, "bold"), relief=tk.FLAT, bd=2)
        self.params_card.pack(fill=tk.BOTH, expand=True)

        self.param_labels = {}
        params = [
            ("File", "file_val", "-"),
            ("Resolution", "res_val", "-"),
            ("Frame", "frame_val", "-"),
            ("FPS", "fps_val", "-"),
            ("Detections", "det_val", "-"),

            ("Confidence", "conf_val", "-"),
        ]

        for i, (label, key, default) in enumerate(params):
            row = tk.Frame(self.params_card, bg=CARD_BG)
            row.pack(fill=tk.X, padx=15, pady=6)
            tk.Label(row, text=label, font=("Segoe UI", 10, "bold"),
                     fg=FG_COLOR, bg=CARD_BG, anchor="w", width=12).pack(side=tk.LEFT)
            lbl = tk.Label(row, text=default, font=("Segoe UI", 10),
                           fg=ACCENT, bg=CARD_BG, anchor="e")
            lbl.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.param_labels[key] = lbl

        # Confidence slider
        slider_frame = tk.Frame(self.params_card, bg=CARD_BG)
        slider_frame.pack(fill=tk.X, padx=15, pady=(15, 15))

        tk.Label(slider_frame, text="Confidence Threshold", font=("Segoe UI", 9),
                 fg=FG_COLOR, bg=CARD_BG).pack(anchor=tk.W)

        scale = tk.Scale(slider_frame, from_=0.1, to=0.9, resolution=0.05,
                         orient=tk.HORIZONTAL, variable=self.conf_threshold,
                         bg=CARD_BG, fg=FG_COLOR, troughcolor=BTN_BG,
                         activebackground=ACCENT, highlightthickness=0,
                         font=("Segoe UI", 8), length=220)
        scale.pack(fill=tk.X, pady=(5, 0))

        self.param_labels["conf_val"].config(text=f"{self.conf_threshold.get():.2f}")
        self.conf_threshold.trace_add("write", lambda *_: self.param_labels["conf_val"].config(
            text=f"{self.conf_threshold.get():.2f}"))

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")]
        )
        if not path:
            return

        self.stop()
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.param_labels["file_val"].config(text="Error opening file")
            return

        self.frame_count = 0
        self.detection_count = 0
        self.processed_count = 0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        fname = os.path.basename(path)
        if len(fname) > 25:
            fname = fname[:22] + "..."

        self.param_labels["file_val"].config(text=fname)
        self.param_labels["res_val"].config(text=f"{w}x{h}")
        self.param_labels["frame_val"].config(text=f"0 / {total_frames}")

        self.play_btn.config(state=tk.NORMAL, text="Play")
        self.stop_btn.config(state=tk.NORMAL)
        self.is_playing = False

        self.show_blank_frame()

    def show_blank_frame(self):
        img = Image.new("RGB", (640, 480), (20, 20, 30))
        self.tk_img = ImageTk.PhotoImage(img)
        self.video_label.config(image=self.tk_img)

    def toggle_play(self):
        if self.cap is None:
            return
        self.is_playing = not self.is_playing
        self.play_btn.config(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.last_time = time.time()
            self.play_frame()

    def play_frame(self):
        if not self.is_playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.is_playing = False
            self.play_btn.config(text="Play")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.param_labels["frame_val"].config(text="0 / -")
            return

        self.frame_count += 1
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        curr = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.param_labels["frame_val"].config(text=f"{curr} / {total}")

        now = time.time()
        self.current_fps = 1.0 / (now - self.last_time + 1e-6)
        self.last_time = now
        self.param_labels["fps_val"].config(text=f"{self.current_fps:.1f}")

        if self.frame_count % self.frame_skip == 0:
            results = self.model(frame, conf=self.conf_threshold.get(), verbose=False)
            self.processed_count += 1
            det_count = 0
            for r in results:
                for box in r.boxes:
                    det_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Pothole {conf*100:.1f}%"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.param_labels["det_val"].config(text=str(det_count))
            if det_count > 0:
                self.detection_count += det_count


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cw = self.video_container.winfo_width() - 20
        ch = self.video_container.winfo_height() - 40
        cw = max(cw, 200)
        ch = max(ch, 100)

        h0, w0 = frame_rgb.shape[:2]
        scale = min(cw / w0, ch / h0, 1.0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        if new_w < 1:
            new_w = 1
        if new_h < 1:
            new_h = 1
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_resized)
        self.tk_img = ImageTk.PhotoImage(img)
        self.video_label.config(image=self.tk_img)

        delay = max(1, int(1000 / (self.current_fps + 1)))
        self.after_id = self.root.after(delay, self.play_frame)

    def stop(self):
        self.is_playing = False
        self.play_btn.config(text="Play")
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.show_blank_frame()
        for key in ["file_val", "res_val", "frame_val", "det_val"]:
            self.param_labels[key].config(text="-")
        self.param_labels["fps_val"].config(text="-")

    def update_stats(self):
        self.root.after(500, self.update_stats)

    def quit(self):
        self.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()
