# -*- coding: utf-8 -*-
"""手検出と色判定を統合した Chewing Sound AR 試作版。

【必要なライブラリ】
    pip install opencv-python sounddevice numpy scipy pillow mediapipe

【実行方法】
    python chewing_sound_ar_fixed.py

"""

from collections import Counter, deque
from pathlib import Path
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from PIL import Image, ImageTk
from scipy.signal import butter, lfilter
import tkinter as tk
from tkinter import messagebox, ttk


SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
HAND_STABLE_SECONDS = 0.4
HAND_MOVEMENT_THRESHOLD = 0.015
NO_HAND_RESET_FRAMES = 3
VOTE_WINDOW = 7
VOTES_TO_CONFIRM = 5


def make_distortion_curve(amount, n=2048):
    x = np.linspace(-1, 1, n, dtype=np.float32)
    if amount <= 0:
        return x.copy()
    deg = np.pi / 180.0
    curve = ((3 + amount) * x * 20 * deg) / (
        np.pi + amount * np.abs(x)
    )
    max_abs = np.max(np.abs(curve))
    if max_abs > 0:
        curve = curve / max_abs
    return curve.astype(np.float32)


def apply_waveshaper(signal, curve):
    n = len(curve)
    idx = np.clip(
        (signal + 1.0) * 0.5 * (n - 1), 0, n - 1
    ).astype(np.int32)
    return curve[idx]


class AudioEngine:
    """マイク入力へエフェクトをかけ、咀嚼中だけヘッドホンへ返す。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.effect = "lowpass"
        self.volume = 0.25
        self.threshold = 0.025

        self.is_chewing = False
        self.volume_percent = 0.0
        self._gain = 0.0
        self._noise_gain = 0.0
        self._prev_high_energy = 0.0

        self._filter_b = self._filter_a = None
        self._filter_zi = None
        self._delay_buf = np.zeros(1, dtype=np.float32)
        self._delay_pos = 0
        self._curve = make_distortion_curve(0)

        rng = np.random.default_rng()
        self._noise = rng.uniform(
            -1, 1, SAMPLE_RATE * 2
        ).astype(np.float32)
        self._noise_pos = 0
        nb, na = butter(
            2, 5000 / (SAMPLE_RATE / 2), btype="highpass"
        )
        self._noise_b, self._noise_a = nb, na
        self._noise_zi = np.zeros(
            max(len(na), len(nb)) - 1, dtype=np.float64
        )
        self.set_effect("lowpass")

    def set_effect(self, effect):
        with self.lock:
            self.effect = effect
            if effect == "lowpass":
                b, a = self._design_filter("lowpass", freq=500)
                delay_t, amount = 0.045, 20
            elif effect == "highpass":
                b, a = self._design_filter("highpass", freq=3500)
                delay_t, amount = 0.005, 80
            else:
                b, a = self._design_filter("bandpass", freq=1300)
                delay_t, amount = 0.02, 400

            self._filter_b, self._filter_a = b, a
            self._filter_zi = np.zeros(
                max(len(a), len(b)) - 1, dtype=np.float64
            )
            self._delay_buf = np.zeros(
                max(int(delay_t * SAMPLE_RATE), 1), dtype=np.float32
            )
            self._delay_pos = 0
            self._curve = make_distortion_curve(amount)

    def set_volume(self, value):
        with self.lock:
            self.volume = float(value)

    def set_threshold(self, value):
        with self.lock:
            self.threshold = float(value)

    @staticmethod
    def _design_filter(kind, freq=700):
        nyq = SAMPLE_RATE / 2.0
        freq = min(max(freq, 20), nyq - 100)
        if kind == "lowpass":
            return butter(2, freq / nyq, btype="lowpass")
        if kind == "highpass":
            return butter(2, freq / nyq, btype="highpass")
        low = max(freq * 0.7, 20) / nyq
        high = min(freq * 1.3, nyq - 50) / nyq
        return butter(2, [low, high], btype="bandpass")

    def _push_delay(self, block):
        buf = self._delay_buf
        out = np.empty_like(block)
        pos = self._delay_pos
        for i, sample in enumerate(block):
            out[i] = buf[pos]
            buf[pos] = sample
            pos += 1
            if pos >= len(buf):
                pos = 0
        self._delay_pos = pos
        return out

    def process(self, indata):
        with self.lock:
            effect = self.effect
            volume = self.volume
            threshold = self.threshold
            x = np.asarray(indata, dtype=np.float32)

            rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
            spec = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)

            mid = spec[(freqs >= 300) & (freqs <= 3000)]
            high = spec[(freqs >= 3000) & (freqs <= 10000)]
            mid_energy = float(np.mean(mid)) if mid.size else 0.0
            high_energy = float(np.mean(high)) if high.size else 0.0
            high_flux = max(
                0.0, high_energy - self._prev_high_energy
            )
            self._prev_high_energy = high_energy

            is_chewing = (
                rms > threshold
                and high_energy / (mid_energy + 1e-6) > 0.35
                and (high_flux > 0.4 or high_energy > 2.0)
            )

            filtered, self._filter_zi = lfilter(
                self._filter_b,
                self._filter_a,
                x,
                zi=self._filter_zi,
            )
            delayed = self._push_delay(filtered.astype(np.float32))
            shaped = apply_waveshaper(
                np.clip(delayed, -1, 1), self._curve
            )

            if is_chewing:
                self._gain = volume
            else:
                self._gain *= 0.8
                if self._gain < 0.001:
                    self._gain = 0.0

            output = shaped * self._gain

            if is_chewing and effect == "highpass":
                target_noise = 0.025
            elif is_chewing and effect == "distortion":
                target_noise = 0.015
            else:
                target_noise = 0.0

            if target_noise > 0:
                self._noise_gain = target_noise
            else:
                self._noise_gain *= 0.8
                if self._noise_gain < 0.0005:
                    self._noise_gain = 0.0

            if self._noise_gain > 0:
                n = len(x)
                idxs = (
                    self._noise_pos + np.arange(n)
                ) % len(self._noise)
                noise_chunk = self._noise[idxs]
                self._noise_pos = (
                    self._noise_pos + n
                ) % len(self._noise)
                noise_f, self._noise_zi = lfilter(
                    self._noise_b,
                    self._noise_a,
                    noise_chunk,
                    zi=self._noise_zi,
                )
                output += (
                    noise_f.astype(np.float32) * self._noise_gain
                )

            self.is_chewing = is_chewing
            self.volume_percent = min(rms * 1000.0, 100.0)

        # 急な過大音を滑らかに抑える簡易リミッター
        return np.tanh(output).astype(np.float32)


def classify_snack_color(crop_bgr, object_mask):
    """背景差分と肌色除外後の画素だけを使って分類する。"""
    if crop_bgr.size == 0 or object_mask.size == 0:
        return "未認識", None

    valid_pixels = int(np.count_nonzero(object_mask))
    if valid_pixels < 120:
        return "未認識", None

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[object_mask > 0]
    hue, saturation, value = np.median(pixels, axis=0)

    # 会場照明と実物に合わせて最終調整する閾値
    if saturation < 55 and value > 135:
        return "まんじゅう", "highpass"
    if 5 <= hue <= 35 and 30 <= saturation <= 180 and value > 60:
        return "せんべい", "lowpass"
    if saturation >= 65 and value > 50:
        return "グミ", "distortion"
    return "未認識", None


def recognize_food_near_fingers(
    frame_bgr, landmarks, background_bgr
):
    """親指と人差し指の間にある、背景・肌以外の領域を判定する。"""
    h, w = frame_bgr.shape[:2]
    thumb = landmarks[4]
    index = landmarks[8]
    wrist = landmarks[0]
    middle_base = landmarks[9]

    cx = int((thumb.x + index.x) * 0.5 * w)
    cy = int((thumb.y + index.y) * 0.5 * h)
    hand_size = np.hypot(
        (wrist.x - middle_base.x) * w,
        (wrist.y - middle_base.y) * h,
    )
    radius = max(35, int(hand_size * 0.8))

    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(w, cx + radius), min(h, cy + radius)
    crop = frame_bgr[y0:y1, x0:x1]

    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if crop.size == 0 or background_bgr is None:
        return "未認識", None

    background_crop = background_bgr[y0:y1, x0:x1]
    if background_crop.shape != crop.shape:
        return "未認識", None

    difference = cv2.absdiff(crop, background_crop)
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    motion_mask = cv2.threshold(
        difference_gray, 25, 255, cv2.THRESH_BINARY
    )[1]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(
        hsv,
        np.array([0, 20, 45], dtype=np.uint8),
        np.array([25, 180, 255], dtype=np.uint8),
    )

    # 高彩度の赤いグミまで肌として消さないように戻す
    vivid_mask = cv2.inRange(
        hsv,
        np.array([0, 181, 40], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    non_skin_mask = cv2.bitwise_or(
        cv2.bitwise_not(skin_mask), vivid_mask
    )
    object_mask = cv2.bitwise_and(motion_mask, non_skin_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    object_mask = cv2.morphologyEx(
        object_mask, cv2.MORPH_OPEN, kernel
    )
    object_mask = cv2.morphologyEx(
        object_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    object_ratio = np.count_nonzero(object_mask) / object_mask.size
    if object_ratio < 0.035:
        return "未認識", None

    return classify_snack_color(crop, object_mask)


class ChewingSoundApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chewing Sound AR Prototype")
        self.engine = AudioEngine()

        model_path = (
            Path(__file__).resolve().parent
            / "models"
            / "hand_landmarker.task"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"手検出モデルが見つかりません: {model_path}"
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.hand_landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )

        self.cap = None
        self.stream = None
        self.running = False
        self.is_locked = False
        self.manual_override = False
        self.last_auto_effect = None

        self.last_hand_center = None
        self.hand_stable_since = None
        self.no_hand_frames = 0
        self.background_frame = None
        self.recognition_history = deque(maxlen=VOTE_WINDOW)
        self.confirmed_label = None
        self.confirmed_effect = None

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)
        ttk.Label(
            top,
            text=(
                "お菓子を手に取って静止してください。"
                "認識後、咀嚼音へエフェクトをかけます。"
            ),
            wraplength=560,
        ).pack(anchor="w")
        self.start_btn = ttk.Button(
            top, text="開始する", command=self.start
        )
        self.start_btn.pack(anchor="w", pady=4)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, **pad)
        left = ttk.Frame(body)
        left.pack(side="left", padx=10)
        self.video_label = ttk.Label(
            left, text="(カメラ映像)", width=40
        )
        self.video_label.pack()
        self.lock_btn = ttk.Button(
            left, text="認識を固定する", command=self.toggle_lock
        )
        self.lock_btn.pack(pady=4)
        self.food_label_var = tk.StringVar(value="未認識")
        ttk.Label(
            left,
            textvariable=self.food_label_var,
            font=("", 12, "bold"),
            wraplength=360,
        ).pack()

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=10)
        ttk.Label(right, text="エフェクト選択").pack(anchor="w")
        self.effect_var = tk.StringVar(value="lowpass")
        effect_box = ttk.Combobox(
            right,
            textvariable=self.effect_var,
            state="readonly",
            values=["lowpass", "highpass", "distortion"],
        )
        effect_box.pack(anchor="w", fill="x")
        effect_box.bind("<<ComboboxSelected>>", self.on_effect_change)
        ttk.Button(
            right,
            text="自動選択に戻す",
            command=self.enable_auto_effect,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(
            right, text="返す音の大きさ (0～1)"
        ).pack(anchor="w", pady=(12, 0))
        self.volume_var = tk.DoubleVar(value=0.25)
        ttk.Scale(
            right,
            from_=0,
            to=1,
            variable=self.volume_var,
            command=lambda value: self.engine.set_volume(value),
        ).pack(fill="x")

        ttk.Label(
            right, text="咀嚼判定の感度 (0.005～0.12)"
        ).pack(anchor="w", pady=(12, 0))
        self.threshold_var = tk.DoubleVar(value=0.025)
        ttk.Scale(
            right,
            from_=0.005,
            to=0.12,
            variable=self.threshold_var,
            command=lambda value: self.engine.set_threshold(value),
        ).pack(fill="x")

        ttk.Label(right, text="咀嚼判定").pack(
            anchor="w", pady=(12, 0)
        )
        self.chew_status_var = tk.StringVar(value="停止中")
        ttk.Label(
            right,
            textvariable=self.chew_status_var,
            font=("", 14, "bold"),
        ).pack(anchor="w")
        self.volume_bar = ttk.Progressbar(right, maximum=100)
        self.volume_bar.pack(fill="x", pady=6)

    def on_effect_change(self, _event=None):
        self.manual_override = True
        self.engine.set_effect(self.effect_var.get())

    def enable_auto_effect(self):
        self.manual_override = False
        if self.confirmed_effect is not None:
            self.effect_var.set(self.confirmed_effect)
            self.engine.set_effect(self.confirmed_effect)

    def toggle_lock(self):
        self.is_locked = not self.is_locked
        if self.is_locked:
            self.lock_btn.config(text="固定を解除する")
        else:
            self.lock_btn.config(text="認識を固定する")
            self._reset_recognition()

    def start(self):
        if self.running:
            return
        self.running = True
        self.start_btn.config(state="disabled", text="実行中…")
        try:
            self.cap = cv2.VideoCapture(
                0, cv2.CAP_AVFOUNDATION
            )
            if not self.cap.isOpened():
                raise RuntimeError(
                    "カメラを開けません。カメラ権限を確認してください。"
                )

            self.stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            self.running = False
            self._close_devices()
            self.start_btn.config(state="normal", text="開始する")
            messagebox.showerror("開始できません", str(exc))
            return

        self._update_video()
        self._update_status()

    def _audio_callback(
        self, indata, outdata, frames, time_info, status
    ):
        del frames, time_info, status
        try:
            outdata[:, 0] = self.engine.process(indata[:, 0])
        except Exception:
            outdata.fill(0)

    def detect_hand(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        result = self.hand_landmarker.detect_for_video(
            mp_image, time.monotonic_ns() // 1_000_000
        )
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def is_hand_stable(self, landmarks):
        now = time.monotonic()
        center = np.array(
            [
                (landmarks[0].x + landmarks[9].x) * 0.5,
                (landmarks[0].y + landmarks[9].y) * 0.5,
            ]
        )
        if self.last_hand_center is None:
            self.last_hand_center = center
            self.hand_stable_since = now
            return False

        movement = np.linalg.norm(center - self.last_hand_center)
        self.last_hand_center = center
        if movement > HAND_MOVEMENT_THRESHOLD:
            self.hand_stable_since = now
            self.recognition_history.clear()
            return False
        if self.hand_stable_since is None:
            self.hand_stable_since = now
            return False
        return now - self.hand_stable_since >= HAND_STABLE_SECONDS

    @staticmethod
    def _draw_hand(frame, landmarks):
        h, w = frame.shape[:2]
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (255, 180, 0), -1)

    def _record_vote(self, label, effect):
        if label == "未認識" or effect is None:
            return False
        self.recognition_history.append((label, effect))
        counts = Counter(item[0] for item in self.recognition_history)
        winner, votes = counts.most_common(1)[0]
        if votes < VOTES_TO_CONFIRM:
            return False

        for recorded_label, recorded_effect in reversed(
            self.recognition_history
        ):
            if recorded_label == winner:
                self.confirmed_label = recorded_label
                self.confirmed_effect = recorded_effect
                break

        if (
            not self.manual_override
            and self.confirmed_effect != self.last_auto_effect
        ):
            self.last_auto_effect = self.confirmed_effect
            self.effect_var.set(self.confirmed_effect)
            self.engine.set_effect(self.confirmed_effect)
        return True

    def _reset_recognition(self):
        self.last_hand_center = None
        self.hand_stable_since = None
        self.no_hand_frames = 0
        self.recognition_history.clear()
        self.confirmed_label = None
        self.confirmed_effect = None

    def _update_video(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if ok:
            if not self.is_locked:
                landmarks = self.detect_hand(frame)
                if landmarks is None:
                    self.no_hand_frames += 1
                    self.background_frame = frame.copy()
                    self.food_label_var.set(
                        "お菓子を手に取ってください"
                    )
                    if self.no_hand_frames >= NO_HAND_RESET_FRAMES:
                        self._reset_recognition()
                else:
                    self.no_hand_frames = 0
                    self._draw_hand(frame, landmarks)

                    if self.confirmed_label is not None:
                        self.food_label_var.set(
                            f"認識結果: {self.confirmed_label}"
                        )
                    elif not self.is_hand_stable(landmarks):
                        self.food_label_var.set(
                            "お菓子を持った手を静止してください"
                        )
                    else:
                        label, effect = recognize_food_near_fingers(
                            frame,
                            landmarks,
                            self.background_frame,
                        )
                        if label == "未認識":
                            self.food_label_var.set(
                                "お菓子を指先の枠内に見せてください"
                            )
                        elif self._record_vote(label, effect):
                            self.food_label_var.set(
                                f"認識結果: {self.confirmed_label}"
                            )
                        else:
                            self.food_label_var.set(
                                f"判定中: {label}"
                            )

            disp = cv2.resize(frame, (360, 270))
            disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(
                image=Image.fromarray(disp_rgb)
            )
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk, text="")

        self.root.after(150, self._update_video)

    def _update_status(self):
        if not self.running:
            return
        self.chew_status_var.set(
            "咀嚼中" if self.engine.is_chewing else "待機中"
        )
        self.volume_bar["value"] = self.engine.volume_percent
        self.root.after(60, self._update_status)

    def _close_devices(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def on_close(self):
        self.running = False
        self._close_devices()
        try:
            self.hand_landmarker.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        app = ChewingSoundApp(root)
    except Exception as exc:
        messagebox.showerror("起動できません", str(exc))
        root.destroy()
        return
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
