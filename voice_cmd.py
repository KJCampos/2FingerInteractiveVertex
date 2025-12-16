# voice_cmd.py
import os
import json
import time
import queue
import threading

import sounddevice as sd
from vosk import Model, KaldiRecognizer


class VoiceCommands:
    """
    Offline voice command recognizer (Vosk).
    Wake-word required: "robin"

    Usage:
        voice = VoiceCommands()
        voice.start()

        # each frame:
        for cmd in voice.poll(max_items=4):
            hud.apply_voice(cmd)

        # on exit:
        voice.stop()
    """

    WAKE_WORDS = ["robin"]

    COMMANDS = [
        "clear", "reset",
        "line", "curve", "grab", "solid",
        "3d", "three d",
        "grid on", "grid off",
        "axis on", "axis off",
        "menu on", "menu off",
        "wire", "neon",
    ]

    def __init__(
        self,
        model_path=None,
        sample_rate=16000,
        device=None,              # set to an int mic index if needed
        require_wake=True,
        arm_window=2.0,           # seconds after wake word to accept commands
        emit_cooldown=0.30,       # seconds to prevent duplicate triggers
        blocksize=1600,           # smaller = lower latency (800–2000 is good)
        debug_print=True,
    ):
        self.sample_rate = int(sample_rate)
        self.device = device
        self.require_wake = bool(require_wake)
        self.arm_window = float(arm_window)
        self.emit_cooldown = float(emit_cooldown)
        self.blocksize = int(blocksize)
        self.debug_print = bool(debug_print)

        # Resolve model path (absolute, stable)
        if model_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(here, "models", "vosk-model-small-en-us-0.15")
        self.model_path = model_path

        # Validate model folder
        if not os.path.isdir(self.model_path):
            raise RuntimeError(f"Vosk model folder not found: {self.model_path}")

        # Optional: sanity check expected subfolders
        must_have = ["am", "conf", "graph"]
        missing = [p for p in must_have if not os.path.exists(os.path.join(self.model_path, p))]
        if missing:
            raise RuntimeError(
                f"Vosk model folder missing {missing}. "
                f"Your folder layout is wrong: {self.model_path}"
            )

        if self.debug_print:
            print("VOSK model path:", self.model_path, "exists?", os.path.isdir(self.model_path))
            print("VOSK model contents:", os.listdir(self.model_path)[:10])

        # Build recognizer with a constrained vocab INCLUDING wake + commands
        vocab = list(dict.fromkeys(self.WAKE_WORDS + self.COMMANDS))
        self.model = Model(self.model_path)
        self.rec = KaldiRecognizer(self.model, self.sample_rate, json.dumps(vocab))

        # Queues + thread control
        self._audio_q = queue.Queue()
        self._cmd_q = queue.Queue()
        self._stop = threading.Event()
        self._thread = None

        # Wake/emit state
        self._armed_until = 0.0
        self._last_emit_t = 0.0
        self._last_text = ""

    def start(self):
        """Start mic capture + recognition thread."""
        if self._thread and self._thread.is_alive():
            return

        def callback(indata, frames, time_info, status):
            if status and self.debug_print:
                print("mic status:", status)
            self._audio_q.put(bytes(indata))

        def worker():
            try:
                # Print devices once if debugging
                if self.debug_print:
                    try:
                        print("voice devices:", sd.query_devices())
                        print("voice default device:", sd.default.device)
                    except Exception as e:
                        print("voice device query failed:", e)

                with sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.blocksize,
                    dtype="int16",
                    channels=1,
                    device=self.device,
                    callback=callback,
                ):
                    if self.debug_print:
                        print("✅ Voice thread started (wake word: robin)")

                    while not self._stop.is_set():
                        data = self._audio_q.get()
                        if self.rec.AcceptWaveform(data):
                            res = json.loads(self.rec.Result())
                            self._handle_text(res.get("text", ""))
                        else:
                            pres = json.loads(self.rec.PartialResult())
                            self._handle_text(pres.get("partial", ""))
            except Exception as e:
                # If thread dies, you see it
                print("❌ Voice thread error:", e)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop recognition thread."""
        self._stop.set()

    def poll(self, max_items=4):
        """Non-blocking fetch of recognized commands."""
        out = []
        for _ in range(max_items):
            try:
                out.append(self._cmd_q.get_nowait())
            except queue.Empty:
                break
        return out

    # ---------------- internal ----------------

    def _handle_text(self, text: str):
        text = (text or "").strip().lower()
        if not text:
            return

        now = time.time()

        # --- wake handling (debounced) ---
        if "robin" in text:
            # debounce wake spam
            last_wake = getattr(self, "_last_wake_t", 0.0)
            if now - last_wake > 0.6:
                self._armed_until = now + self.arm_window
                self._last_wake_t = now
                if self.debug_print:
                    print("🎙️ armed (robin)")

            # IMPORTANT: keep parsing the same phrase for commands too
            text = text.replace("robin", "").strip()
            if not text:
                return

        # Require wake: ignore commands unless armed
        if self.require_wake and now > self._armed_until:
            return

        # Cooldown to prevent repeats
        if now - self._last_emit_t < self.emit_cooldown:
            return

        # Find a command contained in recognized text
        for cmd in self.COMMANDS:
            if cmd in text:
                self._cmd_q.put(cmd)
                self._last_emit_t = now
                self._last_text = text
                if self.debug_print:
                    print("🎙️ voice->cmd:", cmd, "| heard:", text)
                return
