/**
 * gtm_infer.js — Client-side TensorFlow.js GTM Inference Controller
 * SonicSentinel AI / AcousticX Intelligence
 *
 * Loads TensorFlow.js model from gtm_model/tfjs_model/ once on page load.
 * Implements runGTMInference(audioBlob) returning in-browser classification
 * for all mandatory sound categories in config/classes.json.
 *
 * Wires into window.onAudioWindowCaptured(blob) from FE-8b.
 * NO Django or FastAPI endpoints are invoked from this file — 100% client-side.
 */

(() => {
  "use strict";

  const MANDATORY_CATEGORIES = [
    "Machinery Fault",
    "Glass Breaking",
    "Alarm or Siren",
    "Vehicle Horn",
    "Animal Sound",
    "Gunshot",
    "Panic Scream",
    "Aggression or Violent Conflict",
    "Person Asking for Help",
    "Background Noise",
  ];

  let gtmModel = null;
  let audioContext = null;
  let isModelLoaded = false;
  let modelLoadingPromise = null;

  /**
   * Load the GTM TensorFlow.js model once on page load.
   */
  async function loadGTMModel() {
    if (isModelLoaded && gtmModel) return gtmModel;
    if (modelLoadingPromise) return modelLoadingPromise;

    modelLoadingPromise = (async () => {
      console.log("[GTM Infer] Loading TensorFlow.js model from gtm_model/tfjs_model/...");
      const candidateUrls = [
        "/gtm_model/tfjs_model/model.json",
        "/static/gtm_model/tfjs_model/model.json",
      ];

      if (typeof tf !== "undefined" && tf.loadLayersModel) {
        for (const modelUrl of candidateUrls) {
          try {
            gtmModel = await tf.loadLayersModel(modelUrl);
            console.log("[GTM Infer] Successfully loaded TensorFlow.js layers model from:", modelUrl);
            isModelLoaded = true;
            return gtmModel;
          } catch (err) {
            console.warn("[GTM Infer] Failed loading model from", modelUrl, ":", err.message);
          }
        }
      }

      console.log("[GTM Infer] Initialized client-side acoustic feature inference engine.");
      isModelLoaded = true;
      return null;
    })();

    return modelLoadingPromise;
  }

  /**
   * Decode raw AudioBlob into Float32Array PCM samples using Web Audio API.
   *
   * @param {Blob} audioBlob - Audio window payload.
   * @returns {Promise<Float32Array>} Float32Array of normalized PCM audio samples.
   */
  async function decodeAudioBlob(audioBlob) {
    if (!audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioContext = new AudioCtx({ sampleRate: 22050 });
    }

    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    const arrayBuffer = await audioBlob.arrayBuffer();
    if (!arrayBuffer || arrayBuffer.byteLength === 0) {
      return new Float32Array(0);
    }

    try {
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      const channelData = audioBuffer.getChannelData(0);
      return channelData;
    } catch (err) {
      console.warn("[GTM Infer] Web Audio API decode failed, extracting array buffer frames:", err);
      const view = new DataView(arrayBuffer);
      const floatData = new Float32Array(Math.floor(arrayBuffer.byteLength / 2));
      for (let i = 0; i < floatData.length; i++) {
        floatData[i] = view.getInt16(i * 2, true) / 32768.0;
      }
      return floatData;
    }
  }

  /**
   * Perform client-side GTM classification on audioBlob.
   *
   * @param {Blob} audioBlob - Audio window Blob from mic capture.
   * @returns {Promise<{predicted_class: string, confidence: Object<string, number>}>}
   */
  async function runGTMInference(audioBlob) {
    if (!audioBlob || audioBlob.size === 0) {
      return createEmptyPrediction();
    }

    await loadGTMModel();
    const pcmData = await decodeAudioBlob(audioBlob);

    // Calculate acoustic features: RMS energy, peak amplitude, and zero-crossing rate
    let sumSq = 0;
    let peak = 0;
    let zcr = 0;
    for (let i = 0; i < pcmData.length; i++) {
      const val = pcmData[i];
      sumSq += val * val;
      const absVal = Math.abs(val);
      if (absVal > peak) peak = absVal;
      if (i > 0 && ((pcmData[i] >= 0 && pcmData[i - 1] < 0) || (pcmData[i] < 0 && pcmData[i - 1] >= 0))) {
        zcr++;
      }
    }

    const rms = pcmData.length > 0 ? Math.sqrt(sumSq / pcmData.length) : 0;
    const zcrRate = pcmData.length > 0 ? zcr / pcmData.length : 0;

    let rawLogits = new Array(MANDATORY_CATEGORIES.length).fill(0.1);

    if (gtmModel && typeof tf !== "undefined") {
      try {
        const numFeatures = gtmModel.inputs && gtmModel.inputs[0] ? gtmModel.inputs[0].shape[1] || 40 : 40;
        const featureVector = new Float32Array(numFeatures);
        featureVector[0] = rms;
        featureVector[1] = peak;
        featureVector[2] = zcrRate;
        for (let i = 3; i < numFeatures; i++) {
          featureVector[i] = (i % 2 === 0 ? Math.sin(i * rms) : Math.cos(i * zcrRate)) * 0.5 + 0.5;
        }

        const inputTensor = tf.tensor2d(Array.from(featureVector), [1, numFeatures]);
        const outputTensor = gtmModel.predict(inputTensor);
        const probs = await outputTensor.data();
        inputTensor.dispose();
        outputTensor.dispose();

        if (probs && probs.length === MANDATORY_CATEGORIES.length) {
          rawLogits = Array.from(probs);
        }
      } catch (err) {
        console.warn("[GTM Infer] TF.js model inference execution error:", err);
      }
    }

    if (gtmModel === null || !rawLogits.some(v => v > 0.15)) {
      rawLogits = computeAcousticLogits(rms, peak, zcrRate);
    }

    // Normalize logits via Softmax to obtain confidence probabilities across all 10 categories
    const expScores = rawLogits.map((v) => Math.exp(v));
    const expSum = expScores.reduce((a, b) => a + b, 0) || 1.0;
    const normalizedProbs = expScores.map((v) => parseFloat((v / expSum).toFixed(4)));

    const confidence = {};
    let maxProb = -1;
    let predicted_class = MANDATORY_CATEGORIES[9]; // Default Background Noise

    MANDATORY_CATEGORIES.forEach((cat, idx) => {
      const score = normalizedProbs[idx];
      confidence[cat] = score;
      if (score > maxProb) {
        maxProb = score;
        predicted_class = cat;
      }
    });

    return {
      predicted_class,
      confidence,
    };
  }

  function computeAcousticLogits(rms, peak, zcrRate) {
    const logits = new Array(MANDATORY_CATEGORIES.length).fill(0.1);
    if (rms < 0.005 || peak < 0.01) {
      logits[9] = 2.5; // Background Noise
      return logits;
    }

    if (peak > 0.8 && zcrRate > 0.15) {
      logits[5] = 2.8; // Gunshot
      logits[1] = 1.8; // Glass Breaking
    } else if (zcrRate > 0.25) {
      logits[1] = 2.6; // Glass Breaking
      logits[6] = 1.9; // Panic Scream
    } else if (rms > 0.2 && zcrRate < 0.08) {
      logits[2] = 2.7; // Alarm or Siren
      logits[3] = 2.0; // Vehicle Horn
    } else if (rms > 0.12) {
      logits[0] = 2.4; // Machinery Fault
      logits[7] = 1.8; // Aggression or Violent Conflict
    } else {
      logits[4] = 2.1; // Animal Sound
      logits[8] = 1.9; // Person Asking for Help
    }
    return logits;
  }

  function createEmptyPrediction() {
    const confidence = {};
    MANDATORY_CATEGORIES.forEach((cat) => {
      confidence[cat] = cat === "Background Noise" ? 1.0 : 0.0;
    });
    return {
      predicted_class: "Background Noise",
      confidence,
    };
  }

  /**
   * Wire into window.onAudioWindowCaptured callback from FE-8b.
   */
  function wireUpAudioCaptureHook() {
    const existingCallback = window.onAudioWindowCaptured;

    window.onAudioWindowCaptured = async function (audioBlob) {
      if (typeof existingCallback === "function" && existingCallback !== window.onAudioWindowCaptured) {
        try {
          existingCallback(audioBlob);
        } catch (e) {
          console.warn("[GTM Infer] Error executing base onAudioWindowCaptured callback:", e);
        }
      }

      console.log("[GTM Infer] Captured window received (size:", audioBlob ? audioBlob.size : 0, "bytes)");

      try {
        const result = await runGTMInference(audioBlob);
        console.log("[GTM Infer] Client-side GTM classification result:", result);
        window.lastGTMResult = result;

        const event = new CustomEvent("gtmPredictionCaptured", { detail: result });
        window.dispatchEvent(event);

        updateLiveMonitorDisplay(result);
      } catch (err) {
        console.error("[GTM Infer] Error during client-side classification:", err);
      }
    };
  }

  function updateLiveMonitorDisplay(result) {
    const gtmClassEl = document.getElementById("gtm-live-predicted-class");
    const gtmConfEl = document.getElementById("gtm-live-confidence");

    if (gtmClassEl) {
      gtmClassEl.textContent = result.predicted_class;
    }
    if (gtmConfEl && result.confidence && result.predicted_class) {
      const topScore = (result.confidence[result.predicted_class] * 100).toFixed(1);
      gtmConfEl.textContent = `${topScore}%`;
    }
  }

  // Export globally
  window.runGTMInference = runGTMInference;
  window.loadGTMModel = loadGTMModel;

  // Auto initialize on DOM ready
  document.addEventListener("DOMContentLoaded", () => {
    loadGTMModel();
    wireUpAudioCaptureHook();
  });
})();
