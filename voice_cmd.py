# voice_commands.py
import json
import os
import queue
import threading
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer


def _resolve_model_dir(path: str) -> str:
    """
    Handles the common case:
      models/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15/...
    """
    path = os.path.abspath(path)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "am", "final.mdl")):
        return path
    # if a single subfolder contains the real model, pick it
    if os.path.isdir(path):
        subs = [os.path.join(path, d) for d in os.listdir(path)]
        for s in subs:
            if os.path.isdir(s) and os.path.isfile(os.path.join(s, "am", "final.mdl")):
                return s
    return path


class VoiceCommands:
    """
    Wake word gating:
      - Say "robin" to arm for a short window
      - Then say ONE command word/phrase
      - Emits the command text via pop_command()

    Commands emitted are plain strings like:
      "clear", "line", "curve", "wire", "glow", "snap on", "snap off", ...
    """

    def __init__(self,
                 model_path="models/vosk-model-small-en-us-0.15",
                 sample_rate=16000,
                 wake_word="robin",
                 arm_seconds=2.0):
        self.model_path = _resolve_model_dir(model_path)
        self.sample_rate = int(sample_rate)
        self.wake_word = (wake_word or "robin").strip().lower()
        self.arm_seconds = float(arm_seconds)

        if not os.path.isdir(self.model_path):
            raise RuntimeError(f"VOSK model dir not found: {self.model_path}")

        self.model = Model(self.model_path)

        # grammar improves latency + accuracy for small command sets
        self._cmd_phrases = [
            self.wake_word,
            "clear", "reset",
            "line", "curve",
            "wire", "glow", "holo", "debug",
            "snap on", "snap off",
            "axis on", "axis off",
            "grid on", "grid off",
            "mode",
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
                # don’t spam
                return
            self._audio_q.put(bytes(indata))

        def worker():
            # smaller blocksize = less delay
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
                        # partial results can still catch wake word faster
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

    def _handle_text(self, text: str, partial: bool = False):
        now = time.time()

        # If wake word is present anywhere, arm immediately.
        if self.wake_word in text.split():
            self._armed_until = now + self.arm_seconds
            return

        # If not armed, ignore everything else.
        if now > self._armed_until:
            return

        # If partial, don’t emit commands (prevents duplicates/jitter)
        if partial:
            return

        # Find a matching command phrase
        for phrase in self._cmd_phrases:
            if phrase == self.wake_word:
                continue
            # exact phrase match or contained phrase
            if phrase in text:
                self._cmd_q.put(phrase)
                self._armed_until = 0.0  # disarm after 1 command
                return
