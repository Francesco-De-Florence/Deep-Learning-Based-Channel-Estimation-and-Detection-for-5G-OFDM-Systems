# Deep-Learning-Based-Channel-Estimation-and-Detection-for-5G-OFDM-Systems
## Overview

This repository implements **deep learning-based channel estimation for 5G OFDM systems** operating over **multipath Rayleigh fading channels**. The project compares traditional pilot-based channel estimation techniques with modern deep learning models.

Conventional estimators such as **Least Squares (LS)** and **Minimum Mean Square Error (MMSE)** often suffer from degraded performance in noisy and time-varying wireless channels. To address this limitation, this project proposes two neural network-based estimators:

* Residual **Convolutional Neural Network (CNN)**
* **Bidirectional Long Short-Term Memory (BiLSTM)** network

Both models refine the noisy LS channel estimate and improve estimation accuracy across all OFDM subcarriers.

The complete system is implemented in **Python using PyTorch** and simulates an **end-to-end OFDM communication pipeline**.

---

## System Architecture

The implemented communication system follows a standard **5G-inspired OFDM physical layer**.

```
Bit Stream
   │
   ▼
QPSK Modulator
   │
   ▼
Pilot Insertion
   │
   ▼
IFFT (OFDM Modulation)
   │
   ▼
Cyclic Prefix Addition
   │
   ▼
Multipath Rayleigh Channel + AWGN
   │
   ▼
CP Removal + FFT
   │
   ▼
Channel Estimator
(LS / MMSE / CNN / BiLSTM)
   │
   ▼
Zero-Forcing Equalizer
   │
   ▼
QPSK Demodulator
   │
   ▼
Recovered Bits
```

---

## Key Features

* Complete **5G OFDM simulation framework**
* Support for **traditional and deep learning estimators**
* Implementation using **PyTorch**
* Evaluation using **BER and MSE performance metrics**
* Ablation studies on:

  * Pilot density
  * Cyclic prefix length
* Comparison with **perfect CSI performance bound**

---

## Implemented Channel Estimators

### 1. Least Squares (LS)

Simple pilot-based estimator:

[
\hat{H}_{LS}(k) = \frac{Y_k}{X_k}
]

Advantages:

* Very low complexity

Limitations:

* Sensitive to noise
* Poor performance at low SNR

---

### 2. Minimum Mean Square Error (MMSE)

Uses channel correlation information to improve estimation:

[
\hat{H}*{MMSE} = R*{hh}(R_{hh} + \frac{\sigma_w^2}{\sigma_x^2}I)^{-1} \hat{H}_{LS}
]

Advantages:

* Better accuracy than LS

Limitations:

* Requires prior channel statistics
* Higher computational cost

---

### 3. Residual CNN Channel Estimator

A **deep convolutional neural network** designed to refine the LS estimate.

Architecture highlights:

* Encoder–decoder structure
* 1D convolution layers
* Batch normalization
* Residual connection

Residual formulation:

[
\hat{H}*{CNN} = f*\theta(Z) + g_\phi(Z)
]

Where

* (Z = [Re(\hat{H}*{LS}), Im(\hat{H}*{LS})])

Key properties:

* Learns **noise suppression**
* Exploits **frequency correlation between subcarriers**
* Fast convergence during training

---

### 4. Bidirectional LSTM Channel Estimator

Treats the **channel frequency response as a sequential signal** across subcarriers.

Processing steps:

1. Input LS estimate
2. Pass through stacked BiLSTM layers
3. Fully connected regression head outputs CFR

Advantages:

* Captures **long-range subcarrier dependencies**
* Bidirectional processing improves context awareness

---

## Simulation Parameters

| Parameter         | Value           |
| ----------------- | --------------- |
| FFT Size          | 64              |
| Cyclic Prefix     | 16              |
| Pilot Subcarriers | 8               |
| Data Subcarriers  | 48              |
| Guard Subcarriers | 8               |
| Channel Model     | Rayleigh Fading |
| Noise Model       | AWGN            |
| Modulation        | QPSK            |
| Channel Taps      | 6               |
| Training Samples  | 50,000          |
| Test Samples      | 5,000           |
| Optimizer         | AdamW           |
| Epochs            | 60              |

---

## Dataset Generation

Training data is generated using **Monte Carlo simulation**.

For each sample:

1. Random SNR is drawn from **0–30 dB**
2. A new **Rayleigh fading channel** is generated
3. OFDM symbol is transmitted through the channel
4. **LS estimate** is computed
5. **True channel response** is stored as ground truth

Dataset split:

* Training: **50,000 samples**
* Testing: **5,000 samples**

---

## Performance Results

### MSE Comparison

| SNR (dB) | LS     | MMSE   | CNN    | BiLSTM |
| -------- | ------ | ------ | ------ | ------ |
| 0        | 0.18   | 0.09   | 0.04   | 0.03   |
| 10       | 0.062  | 0.029  | 0.012  | 0.009  |
| 20       | 0.020  | 0.009  | 0.0036 | 0.0027 |
| 30       | 0.0068 | 0.0029 | 0.0011 | 0.0008 |

Key observations:

* CNN reduces MSE by **≈60% compared to LS**
* CNN improves **≈35% over MMSE**
* BiLSTM achieves slightly lower MSE at **high SNR**

---

### BER Performance

Deep learning estimators significantly improve BER:

* MMSE achieves **~1.5 dB gain over LS**
* CNN provides **~2 dB gain over MMSE**
* Performance approaches **perfect CSI bound**

---

## Robustness Analysis

### Impact of Pilot Density

Deep learning estimators remain robust even with fewer pilots.

| Pilots | LS MSE | CNN MSE |
| ------ | ------ | ------- |
| 4      | 0.071  | 0.014   |
| 8      | 0.035  | 0.0067  |
| 16     | 0.017  | 0.0027  |

This enables **higher spectral efficiency**.

---

### Impact of Cyclic Prefix Length

When CP is shorter than channel delay spread:

* Inter-symbol interference appears
* Estimation performance degrades

Optimal CP length:

```
Ncp ≥ Maximum channel delay
```

---

## Training Details

Training setup:

* Framework: **PyTorch 2.x**
* GPU: **NVIDIA T4 (Google Colab)**
* Batch size: **256**
* Learning rate: **1e-3**
* Scheduler: **Cosine Annealing**
* Gradient clipping applied for LSTM

Training time:

* CNN: ~12 minutes
* BiLSTM: ~25 minutes

---

## Key Contributions

This project demonstrates:

* Deep learning can **significantly improve OFDM channel estimation**
* CNN and BiLSTM models outperform **LS and MMSE**
* Neural networks are **robust to pilot sparsity**
* Performance approaches **perfect CSI bound**

---

## Future Work

Possible extensions include:

* Transformer-based channel estimators
* Time-varying channel estimation
* Massive MIMO OFDM systems
* Model compression for real-time deployment
* Online learning for adaptive wireless channels

---

## References

1. Dahlman, E., Parkvall, S., Sköld, J. *5G NR: The Next Generation Wireless Access Technology.*
2. Edfors, O. et al. *OFDM Channel Estimation by Singular Value Decomposition.*
3. Ye, H., Li, G. Y. *Deep Learning for Channel Estimation in OFDM Systems.*
