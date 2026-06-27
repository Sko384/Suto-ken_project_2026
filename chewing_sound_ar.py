# -*- coding: utf-8 -*-
"""
Chewing Sound AR Prototype (Python版)
=====================================

元のHTML/JavaScript版を、できるだけ同じロジックでPythonに移植したものです。

【必要なライブラリ】
    pip install opencv-python sounddevice numpy scipy pillow

【実行方法】
    python chewing_sound_ar.py

【注意】
- マイク入力とスピーカー出力を同時に扱う「デュプレックスオーディオ」を使います。
  ヘッドセット(できれば有線)を使うと、スピーカーの音をマイクが拾ってしまう
  ハウリングを防げます。音量は小さめから試してください。
- ブラウザのWeb Audio APIと違い、PythonではOS/オーディオデバイスによって
  遅延(レイテンシ)が変わります。BLOCK_SIZEを小さくすると遅延は減りますが
  CPU負荷が増え、音が途切れる可能性があります。
- カメラによる食べ物判定は元コードと同じく「色だけを見た簡易判定」です。
"""

import threading
import time

import numpy as np
import cv2
import sounddevice as sd
from scipy.signal import butter, lfilter

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


SAMPLE_RATE = 44100
BLOCK_SIZE = 1024  # 1ブロックあたりのサンプル数（小さいほど低遅延・高負荷）


# ----------------------------------------------------------------------
# 音響処理（オリジナルJSの AudioContext / BiquadFilter / WaveShaper に相当）
# ----------------------------------------------------------------------

def make_distortion_curve(amount, n=2048):
    """JS版 makeDistortionCurve() に相当する歪みカーブを作る"""
    x = np.linspace(-1, 1, n)
    if amount <= 0:
        return x.copy()
    deg = np.pi / 180.0
    curve = ((3 + amount) * x * 20 * deg) / (np.pi + amount * np.abs(x))
    max_abs = np.max(np.abs(curve))
    if max_abs > 0:
        curve = curve / max_abs
    return curve.astype(np.float32)


def apply_waveshaper(signal, curve):
    n = len(curve)
    idx = np.clip(((signal + 1.0) / 2.0 * (n - 1)), 0, n - 1).astype(np.int32)
    return curve[idx]


class AudioEngine:
    """マイク入力 -> フィルタ -> ディレイ -> 歪み -> ノイズ付加 -> 出力
    の一連の処理と、咀嚼判定をまとめて行うクラス。
    """

    def __init__(self):
        self.lock = threading.Lock()

        self.effect = "lowpass"
        self.volume = 6.0          # 0..12 出力ゲイン（スライダーと同じ範囲）
        self.threshold = 0.025     # 咀嚼判定の感度(RMS)

        # GUI表示用の状態
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

        # ノイズ源（パリパリ/歪み感の補強用。highpassフィルタ済みホワイトノイズ）
        rng = np.random.default_rng()
        self._noise = rng.uniform(-1, 1, SAMPLE_RATE * 2).astype(np.float32)
        self._noise_pos = 0
        nb, na = butter(2, 5000 / (SAMPLE_RATE / 2), btype="highpass")
        self._noise_b, self._noise_a = nb, na
        self._noise_zi = np.zeros(max(len(na), len(nb)) - 1, dtype=np.float64)

        self.set_effect("lowpass")

    # ---- GUIから呼ばれる設定変更 ----
    def set_effect(self, effect):
        with self.lock:
            self.effect = effect
            if effect == "lowpass":
                b, a = self._design_filter("lowpass", freq=500, q=1.4)
                delay_t, amount = 0.045, 20
            elif effect == "highpass":
                b, a = self._design_filter("highpass", freq=3500, q=2.0)
                delay_t, amount = 0.005, 80
            else:  # distortion
                b, a = self._design_filter("bandpass", freq=1300, q=4.0)
                delay_t, amount = 0.02, 400

            self._filter_b, self._filter_a = b, a
            self._filter_zi = np.zeros(max(len(a), len(b)) - 1, dtype=np.float64)

            new_len = max(int(delay_t * SAMPLE_RATE), 1)
            self._delay_buf = np.zeros(new_len, dtype=np.float32)
            self._delay_pos = 0

            self._curve = make_distortion_curve(amount)

    def set_volume(self, v):
        with self.lock:
            self.volume = float(v)

    def set_threshold(self, t):
        with self.lock:
            self.threshold = float(t)

    @staticmethod
    def _design_filter(kind, freq=700, q=1.0):
        nyq = SAMPLE_RATE / 2.0
        freq = min(max(freq, 20), nyq - 100)
        if kind == "lowpass":
            b, a = butter(2, freq / nyq, btype="lowpass")
        elif kind == "highpass":
            b, a = butter(2, freq / nyq, btype="highpass")
        else:  # bandpass（Qの代わりに帯域幅で近似）
            low = max(freq * 0.7, 20) / nyq
            high = min(freq * 1.3, nyq - 50) / nyq
            b, a = butter(2, [low, high], btype="bandpass")
        return b, a

    def _push_delay(self, block):
        buf = self._delay_buf
        n = len(buf)
        out = np.empty_like(block)
        pos = self._delay_pos
        for i in range(len(block)):
            out[i] = buf[pos]
            buf[pos] = block[i]
            pos += 1
            if pos >= n:
                pos = 0
        self._delay_pos = pos
        return out

    def process(self, indata):
        """1ブロック分のマイク入力(indata: shape=(N,) float32)を処理して
        出力波形(ndarray)を返す。GUI表示用の状態(is_chewing等)も更新する。
        """
        with self.lock:
            effect = self.effect
            volume = self.volume
            threshold = self.threshold

            x = indata.astype(np.float32)

            # --- 咀嚼判定（音量 + 高音成分の割合 + 変化量） ---
            rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
            spec = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)

            def band_energy(lo, hi):
                m = (freqs >= lo) & (freqs <= hi)
                return float(np.mean(spec[m])) if np.any(m) else 0.0

            mid_energy = band_energy(300, 3000)
            high_energy = band_energy(3000, 10000)
            high_flux = max(0.0, high_energy - self._prev_high_energy)
            self._prev_high_energy = high_energy

            is_chewing = (
                rms > threshold
                and (high_energy / (mid_energy + 1e-6) > 0.35)
                and (high_flux > 0.4 or high_energy > 2.0)
            )

            # --- フィルタ -> ディレイ -> 歪み ---
            filtered, self._filter_zi = lfilter(
                self._filter_b, self._filter_a, x, zi=self._filter_zi
            )
            filtered = filtered.astype(np.float32)
            delayed = self._push_delay(filtered)
            shaped = apply_waveshaper(np.clip(delayed, -1, 1), self._curve)

            # --- 出力ゲイン（咀嚼中だけ音を返し、止めたらフェードアウト） ---
            if is_chewing:
                self._gain = volume
            else:
                self._gain *= 0.8
                if self._gain < 0.01:
                    self._gain = 0.0

            output = shaped * self._gain

            # --- ノイズ付加（パリパリ感/歪み感の補強） ---
            if is_chewing and effect == "highpass":
                target_noise = 0.08
            elif is_chewing and effect == "distortion":
                target_noise = 0.035
            else:
                target_noise = 0.0

            if target_noise > 0:
                self._noise_gain = target_noise
            else:
                self._noise_gain *= 0.8
                if self._noise_gain < 0.001:
                    self._noise_gain = 0.0

            if self._noise_gain > 0:
                n = len(x)
                idxs = (self._noise_pos + np.arange(n)) % len(self._noise)
                noise_chunk = self._noise[idxs]
                self._noise_pos = (self._noise_pos + n) % len(self._noise)
                noise_f, self._noise_zi = lfilter(
                    self._noise_b, self._noise_a, noise_chunk, zi=self._noise_zi
                )
                output = output + noise_f.astype(np.float32) * self._noise_gain

            # GUI表示用に保存
            self.is_chewing = is_chewing
            self.volume_percent = min(rms * 1000.0, 100.0)

        return np.clip(output, -1.0, 1.0)


# ----------------------------------------------------------------------
# 食べ物認識（オリジナルJSの recognizeFoodByColor() に相当）
# ----------------------------------------------------------------------

def recognize_food_by_color(frame_bgr):
    """カメラ画像(BGR, numpy配列)の中央下寄りをクロップして、
    平均色から「まんじゅう/せんべい/グミ」を簡易判定する。
    戻り値: (label, effect_name)
    """
    h, w, _ = frame_bgr.shape
    crop_w, crop_h = int(w * 0.4), int(h * 0.4)
    x0 = (w - crop_w) // 2
    y0 = int(h * 0.5)
    y1 = min(y0 + crop_h, h)
    x1 = min(x0 + crop_w, w)

    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return "未認識", "lowpass"

    b, g, r = [float(np.mean(crop[:, :, i])) for i in range(3)]
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    saturation = max_c - min_c

    if saturation < 50 and max_c > 120:
        return "まんじゅう", "highpass"
    elif r > 90 and g > 70 and b < 130:
        return "せんべい", "lowpass"
    else:
        return "グミ", "lowpass"


# ----------------------------------------------------------------------
# GUI（オリジナルJSのHTML/CSS部分に相当）
# ----------------------------------------------------------------------

class ChewingSoundApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chewing Sound AR Prototype (Python)")

        self.engine = AudioEngine()
        self.cap = None
        self.stream = None
        self.running = False
        self.is_locked = False
        self.last_auto_effect = None
        self.manual_override = False  # ユーザーがドロップダウンを直接変えたか

        self._build_ui()

    # ---------------- UI構築 ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)
        ttk.Label(
            top,
            text=(
                "カメラで食べ物を判別 → マイク音から咀嚼を検出 → "
                "イヤホン/スピーカーへエフェクト音を返します"
            ),
            wraplength=520,
        ).pack(anchor="w")

        self.start_btn = ttk.Button(top, text="開始する", command=self.start)
        self.start_btn.pack(anchor="w", pady=4)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, **pad)

        # 左：カメラ映像
        left = ttk.Frame(body)
        left.pack(side="left", padx=10)
        self.video_label = ttk.Label(left, text="(カメラ映像)", width=40)
        self.video_label.pack()

        self.lock_btn = ttk.Button(left, text="認識を固定する", command=self.toggle_lock)
        self.lock_btn.pack(pady=4)

        self.food_label_var = tk.StringVar(value="未認識")
        ttk.Label(left, textvariable=self.food_label_var, font=("", 12, "bold")).pack()

        # 右：コントロール
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

        ttk.Label(right, text="返す音の大きさ (0-12)").pack(anchor="w", pady=(12, 0))
        self.volume_var = tk.DoubleVar(value=6.0)
        ttk.Scale(
            right, from_=0, to=12, variable=self.volume_var,
            command=lambda v: self.engine.set_volume(float(v)),
        ).pack(fill="x")

        ttk.Label(right, text="咀嚼判定の感度 (0.005-0.12)").pack(anchor="w", pady=(12, 0))
        self.threshold_var = tk.DoubleVar(value=0.025)
        ttk.Scale(
            right, from_=0.005, to=0.12, variable=self.threshold_var,
            command=lambda v: self.engine.set_threshold(float(v)),
        ).pack(fill="x")

        ttk.Label(right, text="咀嚼判定").pack(anchor="w", pady=(12, 0))
        self.chew_status_var = tk.StringVar(value="停止中")
        ttk.Label(right, textvariable=self.chew_status_var, font=("", 14, "bold")).pack(anchor="w")

        self.volume_bar = ttk.Progressbar(right, maximum=100)
        self.volume_bar.pack(fill="x", pady=6)

        ttk.Label(
            right,
            text=(
                "音量が一定以上あり、さらに高音成分の割合が大きい時だけ\n"
                "「咀嚼中」と判定します（会話音/環境音への誤反応を減らす版）。"
            ),
            foreground="#555",
            wraplength=320,
        ).pack(anchor="w", pady=(8, 0))

    # ---------------- イベント ----------------
    def on_effect_change(self, _event=None):
        self.manual_override = True
        self.engine.set_effect(self.effect_var.get())

    def toggle_lock(self):
        self.is_locked = not self.is_locked
        if self.is_locked:
            self.lock_btn.config(text="固定を解除する")
        else:
            self.lock_btn.config(text="認識を固定する")

    # ---------------- 開始処理 ----------------
    def start(self):
        if self.running:
            return
        self.running = True
        self.start_btn.config(state="disabled", text="実行中…")

        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()

        self._update_video()
        self._update_status()

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            # バッファのアンダーラン等。デバッグ時はprintしてもよい
            pass
        x = indata[:, 0]
        y = self.engine.process(x)
        outdata[:, 0] = y

    # ---------------- 映像更新ループ（500ms毎に食べ物認識） ----------------
    def _update_video(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            if not self.is_locked:
                label, effect = recognize_food_by_color(frame)
                self.food_label_var.set(
                    label + ("（固定中）" if self.is_locked else "")
                )
                if not self.manual_override and effect != self.last_auto_effect:
                    self.last_auto_effect = effect
                    self.effect_var.set(effect)
                    self.engine.set_effect(effect)

            # プレビュー表示用に縮小してTkinterに渡す
            disp = cv2.resize(frame, (360, 270))
            disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(disp_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk, text="")

        self.root.after(150, self._update_video)

    # ---------------- ステータス表示更新ループ ----------------
    def _update_status(self):
        if not self.running:
            return
        if self.engine.is_chewing:
            self.chew_status_var.set("咀嚼中")
        else:
            self.chew_status_var.set("待機中")
        self.volume_bar["value"] = self.engine.volume_percent
        self.root.after(60, self._update_status)

    # ---------------- 終了処理 ----------------
    def on_close(self):
        self.running = False
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ChewingSoundApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
