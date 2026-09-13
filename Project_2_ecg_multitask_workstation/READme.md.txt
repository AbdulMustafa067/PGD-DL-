# 🩺 Next-Gen Multi-Task AI ECG Diagnostic Workstation

A production-grade Deep Sequence Learning application that processes real-time continuous electrocardiogram (ECG) telemetry signal matrices. It utilizes a **Multi-Task Functional API architecture** to simultaneously classify cardiac pathologies and evaluate autonomic psychological strain from a single sequence stream.

## 🧠 Architectural Overview
Instead of chaining independent networks, this system implements a shared feature extraction backbone to minimize resource overhead at the clinical edge:
* **Spatial Feature Extraction:** Two sequential 1D Convolutional Neural Network (CNN) blocks capture localized micro-oscillations, voltage amplitudes, and physical spike features (P-wave, QRS-complex, ST-segment deviations).
* **Temporal Memory Alignment:** A deep Gated Recurrent Unit/LSTM layer extracts long-term sequence history and heart rate frequency variations over time.
* **Functional Multi-Output Heads:** Dual dedicated dense branches execute classification paths using categorical `softmax` optimization profiles for concurrent output targets.

## 📊 Performance & Task Matrix
1. **Cardiac Condition Branch (3-Class):** Differentiates between Normal Sinus Rhythms, Errhythmia Spikes, and acute Myocardial Infarction Risks (Heart Attack markers).
2. **Autonomic Stress Branch (2-Class):** Extracts R-R peak interval variances to classify physiological anxiety/strain levels into Low or Elevated categories.
3. **Clinical Verification:** Features an interactive Gradio UI containing 10 diverse pre-compiled clinical cases for direct diagnostic validation.

## 🚀 Execution & Setup
Ensure you have Python 3.10+ installed along with the required libraries:
```bash
pip install -r requirements.txt
python src/main_pipeline.py
```
