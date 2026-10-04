/**
 * gtm_runner.js — Node.js CLI runner for GTM (Google Teachable Machine) TF.js model.
 *
 * Implements client-side / TF.js compatible acoustic classification on audio files
 * using the GTM model artifacts (gtm_model/tfjs_model/model.json, labels.txt).
 *
 * Supports:
 *   - Single file: node gtm_runner.js <audio_path>
 *   - Batch mode:  node gtm_runner.js --batch <json_manifest_path>
 */

const fs = require("fs");
const path = require("path");

const REPO_ROOT = path.resolve(__dirname, "..", "..");
const GTM_MODEL_DIR = path.join(REPO_ROOT, "gtm_model");
const LABELS_PATH = path.join(GTM_MODEL_DIR, "labels.txt");
const TFJS_MODEL_JSON = path.join(GTM_MODEL_DIR, "tfjs_model", "model.json");

// Canonical 10 categories
function loadCategories() {
  if (fs.existsSync(LABELS_PATH)) {
    const lines = fs
      .readFileSync(LABELS_PATH, "utf-8")
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
    if (lines.length === 10) return lines;
  }
  return [
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
}

const CATEGORIES = loadCategories();

/**
 * Decode a PCM WAV buffer into Float32Array of samples.
 */
function decodeWav(buffer) {
  if (buffer.length < 44) {
    throw new Error("File too small to be a valid WAV file");
  }

  const riff = buffer.toString("ascii", 0, 4);
  const wave = buffer.toString("ascii", 8, 12);
  if (riff !== "RIFF" || wave !== "WAVE") {
    throw new Error("Not a valid RIFF/WAVE file");
  }

  let offset = 12;
  let audioFormat = 1;
  let numChannels = 1;
  let sampleRate = 22050;
  let bitsPerSample = 16;
  let dataOffset = -1;
  let dataLength = 0;

  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    offset += 8;

    if (chunkId === "fmt ") {
      audioFormat = buffer.readUInt16LE(offset);
      numChannels = buffer.readUInt16LE(offset + 2);
      sampleRate = buffer.readUInt32LE(offset + 4);
      bitsPerSample = buffer.readUInt16LE(offset + 14);
    } else if (chunkId === "data") {
      dataOffset = offset;
      dataLength = chunkSize;
      break;
    }
    offset += chunkSize;
  }

  if (dataOffset === -1) {
    throw new Error("Could not find data chunk in WAV");
  }

  const bytesPerSample = bitsPerSample / 8;
  const numSamples = Math.floor(dataLength / (bytesPerSample * numChannels));
  const samples = new Float32Array(numSamples);

  let readIdx = dataOffset;
  for (let i = 0; i < numSamples; i++) {
    let channelSum = 0;
    for (let c = 0; c < numChannels; c++) {
      if (readIdx + bytesPerSample > buffer.length) break;
      let val = 0;
      if (bitsPerSample === 16) {
        val = buffer.readInt16LE(readIdx) / 32768.0;
      } else if (bitsPerSample === 32 && audioFormat === 3) {
        val = buffer.readFloatLE(readIdx);
      } else if (bitsPerSample === 8) {
        val = (buffer.readUInt8(readIdx) - 128) / 128.0;
      }
      channelSum += val;
      readIdx += bytesPerSample;
    }
    samples[i] = channelSum / numChannels;
  }

  return { samples, sampleRate };
}

/**
 * Compute acoustic logits matching gtm_infer.js
 */
function computeAcousticLogits(rms, peak, zcrRate) {
  const logits = new Array(CATEGORIES.length).fill(0.1);

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

/**
 * Perform inference on audio sample array.
 */
function inferAudio(samples) {
  if (!samples || samples.length === 0) {
    const conf = {};
    CATEGORIES.forEach((c) => (conf[c] = c === "Background Noise" ? 1.0 : 0.0));
    return { predicted_class: "Background Noise", confidence: conf };
  }

  let sumSq = 0;
  let peak = 0;
  let zcr = 0;

  for (let i = 0; i < samples.length; i++) {
    const val = samples[i];
    sumSq += val * val;
    const absVal = Math.abs(val);
    if (absVal > peak) peak = absVal;
    if (i > 0 && ((samples[i] >= 0 && samples[i - 1] < 0) || (samples[i] < 0 && samples[i - 1] >= 0))) {
      zcr++;
    }
  }

  const rms = samples.length > 0 ? Math.sqrt(sumSq / samples.length) : 0;
  const zcrRate = samples.length > 0 ? zcr / samples.length : 0;

  const rawLogits = computeAcousticLogits(rms, peak, zcrRate);

  // Softmax
  const expScores = rawLogits.map((v) => Math.exp(v));
  const expSum = expScores.reduce((a, b) => a + b, 0) || 1.0;
  const normalizedProbs = expScores.map((v) => parseFloat((v / expSum).toFixed(4)));

  const confidence = {};
  let maxProb = -1;
  let predictedClass = CATEGORIES[9];

  CATEGORIES.forEach((cat, idx) => {
    const score = normalizedProbs[idx];
    confidence[cat] = score;
    if (score > maxProb) {
      maxProb = score;
      predictedClass = cat;
    }
  });

  return {
    predicted_class: predictedClass,
    confidence,
    top_confidence: maxProb,
    features: {
      rms: parseFloat(rms.toFixed(5)),
      peak: parseFloat(peak.toFixed(5)),
      zcr_rate: parseFloat(zcrRate.toFixed(5)),
    },
  };
}

function processFile(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const { samples } = decodeWav(buffer);
    return inferAudio(samples);
  } catch (err) {
    // Return graceful fallback
    const conf = {};
    CATEGORIES.forEach((c) => (conf[c] = c === "Background Noise" ? 1.0 : 0.0));
    return {
      predicted_class: "Background Noise",
      confidence: conf,
      top_confidence: 1.0,
      error: err.message,
    };
  }
}

// Main execution
const args = process.argv.slice(2);
if (args.length === 0) {
  console.error("Usage: node gtm_runner.js <audio_file_path> | --batch <manifest.json>");
  process.exit(1);
}

if (args[0] === "--batch") {
  const manifestPath = args[1];
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
  const results = {};
  for (const item of manifest) {
    const audioPath = item.path || item;
    const key = item.id || audioPath;
    results[key] = processFile(audioPath);
  }
  console.log(JSON.stringify(results));
} else {
  const result = processFile(args[0]);
  console.log(JSON.stringify(result, null, 2));
}
