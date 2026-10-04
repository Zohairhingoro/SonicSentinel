/**
 * mic_capture.js — Live Microphone Capture Controller (FE-8b)
 * SonicSentinel AI / AcousticX Intelligence
 *
 * Captures audio in fixed 1-3 second windows (window duration read dynamically
 * from config/audio_config.json via template context value).
 *
 * For each window, packages audio into a Blob and invokes placeholder stub:
 *   onAudioWindowCaptured(blob)
 * Updates FE-8a status widget to "Active" while capturing, and "Available" when stopped.
 */

(() => {
  "use strict";

  /**
   * Placeholder callback function triggered whenever a fixed audio window is captured.
   *
   * STUB FOR FE-9 & BE INTEGRATION:
   * This function will be expanded in FE-9 / BE integration to pass the captured audio
   * chunk blob to client-side GTM (TensorFlow.js) and post to the Python ML inference API (/predict).
   *
   * @param {Blob} blob - Audio chunk blob for the window (1-3s)
   */
  window.onAudioWindowCaptured = window.onAudioWindowCaptured || function onAudioWindowCaptured(blob) {
    console.log('[MicCapture STUB] Captured audio window blob:', blob.size, 'bytes, type:', blob.type);
  };

  class MicCaptureController {
    constructor() {
      this.mediaStream = null;
      this.recordingIntervalId = null;
      this.isRecording = false;

      // DOM Elements
      this.container = document.getElementById("mic-capture-container");
      this.startBtn = document.getElementById("start-mic-btn");
      this.stopBtn = document.getElementById("stop-mic-btn");
      this.durationDisplay = document.getElementById("window-duration-display");

      // Read window duration from container data attribute or global context
      const dataDuration = this.container ? this.container.getAttribute("data-window-duration") : null;
      const parsedDuration = parseFloat(dataDuration || window.AUDIO_WINDOW_DURATION || 3.0);
      this.windowDurationMs = (isNaN(parsedDuration) || parsedDuration <= 0 ? 3.0 : parsedDuration) * 1000;

      this.init();
    }

    init() {
      if (this.startBtn) {
        this.startBtn.addEventListener("click", () => this.startCapture());
      }
      if (this.stopBtn) {
        this.stopBtn.addEventListener("click", () => this.stopCapture());
      }

      if (this.durationDisplay) {
        this.durationDisplay.textContent = (this.windowDurationMs / 1000).toFixed(1) + "s";
      }

      this.updateButtonStates(false);
    }

    async startCapture() {
      if (this.isRecording) return;

      try {
        // Request microphone media stream
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 22050,
            echoCancellation: true,
            noiseSuppression: true,
          },
        });

        // Update FE-8a status widget to Active
        if (window.MicStatus && typeof window.MicStatus.set === "function") {
          window.MicStatus.set("active");
        }

        this.isRecording = true;
        this.updateButtonStates(true);

        // Determine supported audio MIME type
        const mimeType = this.getSupportedMimeType();
        const options = mimeType ? { mimeType } : {};

        // Start capture slicing loop
        this.startSlicingLoop(options);

      } catch (err) {
        console.error("[MicCapture] Error accessing microphone:", err);
        this.isRecording = false;
        this.updateButtonStates(false);

        if (window.MicStatus && typeof window.MicStatus.set === "function") {
          if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            window.MicStatus.set("denied");
          } else {
            window.MicStatus.set("disconnected");
          }
        }
      }
    }

    startSlicingLoop(options) {
      const recordWindow = () => {
        if (!this.isRecording || !this.mediaStream || !this.mediaStream.active) return;

        let recorder = null;
        let chunks = [];

        try {
          recorder = new MediaRecorder(this.mediaStream, options);
        } catch (e) {
          recorder = new MediaRecorder(this.mediaStream);
        }

        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) {
            chunks.push(e.data);
          }
        };

        recorder.onstop = () => {
          if (chunks.length > 0) {
            const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
            if (typeof window.onAudioWindowCaptured === "function") {
              window.onAudioWindowCaptured(blob);
            }
          }
        };

        recorder.start();

        // Stop this window recorder exactly when windowDurationMs expires
        setTimeout(() => {
          if (recorder && recorder.state === "recording") {
            recorder.stop();
          }
        }, this.windowDurationMs);
      };

      // Record initial window immediately
      recordWindow();

      // Schedule continuous window recording
      this.recordingIntervalId = setInterval(() => {
        if (this.isRecording) {
          recordWindow();
        } else {
          clearInterval(this.recordingIntervalId);
        }
      }, this.windowDurationMs);
    }

    stopCapture() {
      this.isRecording = false;

      if (this.recordingIntervalId) {
        clearInterval(this.recordingIntervalId);
        this.recordingIntervalId = null;
      }

      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach((track) => track.stop());
        this.mediaStream = null;
      }

      // Update FE-8a status widget to Available
      if (window.MicStatus && typeof window.MicStatus.set === "function") {
        window.MicStatus.set("available");
      }

      this.updateButtonStates(false);
    }

    updateButtonStates(isCapturing) {
      if (this.startBtn) {
        this.startBtn.disabled = isCapturing;
        if (isCapturing) {
          this.startBtn.classList.add("opacity-50", "cursor-not-allowed");
        } else {
          this.startBtn.classList.remove("opacity-50", "cursor-not-allowed");
        }
      }

      if (this.stopBtn) {
        this.stopBtn.disabled = !isCapturing;
        if (!isCapturing) {
          this.stopBtn.classList.add("opacity-50", "cursor-not-allowed");
        } else {
          this.stopBtn.classList.remove("opacity-50", "cursor-not-allowed");
        }
      }
    }

    getSupportedMimeType() {
      const types = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4",
        "audio/wav",
      ];
      for (const t of types) {
        if (MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) {
          return t;
        }
      }
      return "";
    }
  }

  // Export class & auto-initialize on DOM ready
  window.MicCaptureController = MicCaptureController;

  document.addEventListener("DOMContentLoaded", () => {
    window.micCaptureController = new MicCaptureController();
  });
})();
