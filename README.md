# EEG Signal-Processing & Motor Imagery BCI Pipeline

A Python-based signal processing pipeline designed to simulate multi-channel Electroencephalography (EEG) data, apply digital filtering to isolate sensorimotor rhythms, and extract band-power feature vectors for Brain-Computer Interface (BCI) motor imagery classification.

---

## 📌 Project Overview

Motor imagery—the mental execution of a movement without neuromuscular output—causes localized frequency shifts in the brain's motor cortex known as **Event-Related Desynchronization (ERD)** and **Event-Related Synchronization (ERS)**. Decoding these neural oscillations allows brain-computer interfaces (BCIs) to translate imagined movements into control signals for neural prosthetics and assistive bio-devices.

This project implements an end-to-end signal processing architecture that:
1. **Simulates Multi-Channel EEG Data:** Generates synthetic multi-electrode recordings containing background neurological noise and modulated sensorimotor oscillations.
2. **Isolates Frequency Bands:** Applies a continuous-time to discrete-time Butterworth bandpass filter to isolate key frequency spectra ($\mu$ and $\beta$ rhythms).
3. **Extracts Feature Vectors:** Computes spatial channel band-power (variance) to represent motor intent for downstream classification algorithms.

---

## 🔬 Mathematical & Signal Processing Background

### 1. Sensorimotor Rhythms (SMR)
Motor imagery primarily modulates activity within two primary frequency bands over the sensorimotor cortex:
* **$\mu$ (Mu) Band ($8 - 13 \text{ Hz}$):** Attenuates (ERD) during active motor imagery.
* **$\beta$ (Beta) Band ($13 - 30 \text{ Hz}$):** Exhibits post-imagery rebound (ERS).

### 2. Digital Filtering (Butterworth Bandpass)
To isolate the $8 - 30 \text{ Hz}$ frequency spectrum from high-frequency muscle artifacts (EMG) and low-frequency drift, we utilize an $n^\text{th}$-order digital Butterworth filter. The magnitude response of an $n^\text{th}$-order lowpass Butterworth filter is given by:

$$|H(f)|^2 = \frac{1}{1 + \left(\frac{f}{f_c}\right)^{2n}}$$

where $f_c$ represents the cutoff frequency. Applying this via zero-phase forward-backward digital filtering yields isolated sensorimotor signals without phase distortion.

### 3. Feature Extraction (Channel Band Power)
For each filtered electrode channel $k \in \{1, 2, \dots, C\}$ with $N$ discrete time samples, the signal power is quantified by its sample variance $\sigma_k^2$:

$$\sigma_k^2 = \frac{1}{N} \sum_{n=1}^{N} \left( x_k[n] - \bar{x}_k \right)^2$$

This yields a compact feature vector $\mathbf{v} = [\sigma_1^2, \sigma_2^2, \dots, \sigma_C^2]^T \in \mathbb{R}^C$ ready for pattern classification.

---

## 📁 Repository Structure

```text
eeg-motor-imagery-bci/
├── data/                  # Placeholder for raw EEG datasets (e.g., PhysioNet BCI)
├── src/
│   ├── __init__.py
│   ├── simulator.py       # Synthetic multi-channel EEG generation module
│   ├── filtering.py       # Digital Butterworth filter implementations
│   └── feature_extraction.py # Band-power and variance metrics
├── bci_pipeline.py        # Main execution pipeline script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## ⚡ Quick Start & Installation

### Prerequisites
* Python 3.8 or higher
* `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/eeg-motor-imagery-bci.git
cd eeg-motor-imagery-bci
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

*Dependencies include:* `numpy`, `scipy`, `matplotlib`

### 3. Run the Processing Pipeline
```bash
python bci_pipeline.py
```

---

## 🗺️ Project Roadmap & Future Enhancements

- [x] Synthetic multi-channel EEG data generation ($f_s = 250 \text{ Hz}$).
- [x] Digital Butterworth bandpass filtering ($8 - 30 \text{ Hz}$).
- [x] Spatial band-power feature vector extraction.
- [ ] **Real-world Dataset Integration:** Ingest and preprocess open-access recordings from the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/).
- [ ] **Spatial Filtering:** Implement **Common Spatial Pattern (CSP)** algorithms to maximize feature variance between discrete motor imagery classes (e.g., left hand vs. right hand).
- [ ] **Classification:** Benchmark Support Vector Machine (SVM) and Linear Discriminant Analysis (LDA) classifiers on extracted feature matrices.
