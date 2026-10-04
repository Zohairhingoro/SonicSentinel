# AI Usage Declaration — SonicSentinel AI

> **Competition Compliance Document** — TechWiz7 (Aptech) NextWave AI and ML Competition  
> System: **SonicSentinel AI** (*AcousticX Intelligence*)

In accordance with competition rules and transparency guidelines, all AI coding assistants, generative AI models, and design tools utilized during the research, dataset processing, UI design, and application development of SonicSentinel AI are explicitly declared below.

---

| Date | Tool / Model | Purpose | Files Affected | Manually Reviewed / Modified | Verified By |
|---|---|---|---|---|---|
| 2026-09-24 | **Claude (Anthropic)** | Dataset Sourcing & Sourcing Strategy | `audio_dataset/raw/`, `docs/dev_log.md` | Researched and identified public audio datasets (UrbanSound8K, ESC-50, AudioSet subsets) for the 10 mandatory sound categories. Verified audio licensing, sampling rate compatibility (22.05kHz mono/16kHz), and class distribution integrity. | Dev Team |
| 2026-09-24 | **Antigravity AI (Google)** | Dataset Sorting, Cleaning & Stratified Splitting | `ml_pipeline/data/build_dataset_index.py`, `ml_pipeline/data/stratified_split.py`, `audio_dataset/metadata/dataset_index.csv` | Automated sorting of raw audio clips into canonical class folders (`machinery_fault`, `glass_breaking`, `alarm_or_siren`, etc.), built metadata index CSV, and performed 70% train / 15% val / 15% test stratified split without recording leakage. | Dev Team |
| 2026-09-25 | **Claude (Anthropic)** | UI/UX Concepts, Visual Design & Layout Ideas | `webapp/templates/`, `webapp/static/css/`, `webapp/dashboard/templates/` | Generated UI/UX design concepts, modern dark-mode aesthetic guidelines, color palette selections (Tailored HSL accents, glassmorphic card containers), dashboard grid layouts, and visual hierarchy for operational roles. | Dev Team |

---

### Declaration Guidelines & Integrity Verification

1. **No Hardcoded Predictions**: All prediction outputs are generated dynamically by the trained Python ML model and client-side TensorFlow.js model.
2. **Decoupled Pipelines**: Python server-side model outputs and TensorFlow.js client-side model outputs remain strictly independent.
3. **Manual Code Verification**: Every AI-assisted file was inspected, manually refined, tested, and validated by the development team prior to final deployment.
