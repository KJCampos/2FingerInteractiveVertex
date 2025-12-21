import json
import os
import queue
import threading
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer


def _resolve_model_dir(path: str) -> str:
    """Resolve the real VOSK model directory.

    Rules:
    - If the folder contains a single subfolder that itself has am/, use the subfolder.
    - Otherwise, use the folder directly.
    """
    base = os.path.abspath(path)
    if not os.path.isdir(base):
        return base

    entries = [d for d in os.listdir(base) if not d.startswith('.')]
    if len(entries) == 1:
        candidate = os.path.join(base, entries[0])
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "am")):
            return candidate

    if os.path.isdir(os.path.join(base, "am")):
        return base

    return base


class VoiceCommands:
    """Wake-word gated commands using VOSK."""

    def __init__(self, model_path="models/vosk-model-small-en-us-0.15", sample_rate=16000, wake_word="robin", arm_seconds=2.0):
        self.model_path = _resolve_model_dir(model_path)
        self.sample_rate = int(sample_rate)
        self.wake_word = (wake_word or "robin").strip().lower()
        self.arm_seconds = float(arm_seconds)

        if not os.path.isdir(self.model_path):
            raise RuntimeError(f"VOSK model dir not found: {self.model_path}")

        self.model = Model(self.model_path)

        self._cmd_phrases = [
            self.wake_word,
            "clear",
            "clear all",
            "delete",
            "reset",
            "line",
            "curve",
        ]
        grammar = json.dumps(self._cmd_phrases)
        self.rec = KaldiRecognizer(self.model, self.sample_rate, grammar)

        self._audio_q = queue.Queue()
        self._cmd_q = queue.Queue()
        self._stop = threading.Event()
        self._thread = None

        self._armed_until = 0.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        def callback(indata, frames, t, status):
            if status:
                return
            self._audio_q.put(bytes(indata))

        def worker():
            block = 2000  # ~125ms at 16k
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=block,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                print(f"✅ Voice thread started (wake word: {self.wake_word})")
                while not self._stop.is_set():
                    data = self._audio_q.get()
                    if self.rec.AcceptWaveform(data):
                        res = json.loads(self.rec.Result() or "{}")
                        text = (res.get("text") or "").strip().lower()
                        if text:
                            self._handle_text(text)
                    else:
                        pres = json.loads(self.rec.PartialResult() or "{}")
                        ptxt = (pres.get("partial") or "").strip().lower()
                        if ptxt:
                            self._handle_text(ptxt, partial=True)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def pop_command(self):
        try:
            return self._cmd_q.get_nowait()
        except queue.Empty:
            return None

    # -------- internals --------

    def _arm(self):
        self._armed_until = time.time() + self.arm_seconds

    def _handle_text(self, text: str, partial: bool = False):
        text = text.strip().lower()
        now = time.time()

        # Wake word handling (supports "robin clear" as one phrase)
        if self.wake_word in text.split() or text.startswith(self.wake_word + " "):
            self._arm()
            if text != self.wake_word:
                remaining = text.replace(self.wake_word, "", 1).strip()
                if remaining:
                    self._handle_command(remaining)
            return

        if now > self._armed_until:
            return

        if partial:
            return

        self._handle_command(text)

    def _handle_command(self, text: str):
        for phrase in self._cmd_phrases:
            if phrase == self.wake_word:
                continue
            if phrase == text or phrase in text:
                self._cmd_q.put(phrase)
                self._armed_until = 0.0
                return
