# Deep Learning-Based Channel Estimation for 5G OFDM Systems

## Overview
This project implements deep learning-based channel estimation for 5G Orthogonal Frequency Division Multiplexing (OFDM) systems operating over multipath Rayleigh fading channels. The study compares traditional pilot-based estimators with modern neural network approaches to improve channel estimation accuracy and communication reliability.

Two neural network architectures are investigated:

- Residual Convolutional Neural Network (CNN)
- Bidirectional Long Short-Term Memory (BiLSTM)

Both models refine a noisy Least Squares (LS) channel estimate and aim to approach the performance of perfect Channel State Information (CSI).

## Key Features
- End-to-end OFDM simulation framework
- QPSK modulation and pilot-assisted transmission
- Multipath Rayleigh fading channel model
- Additive White Gaussian Noise (AWGN)
- Comparison of classical and deep learning channel estimators
- Performance evaluation using Mean Square Error (MSE) and Bit Error Rate (BER)
- Ablation analysis for pilot density and cyclic prefix length

## System Architecture

The implemented communication pipeline consists of the following components:

1. OFDM Transmitter
2. Pilot insertion and QPSK mapping
3. IFFT modulation and cyclic prefix addition
4. Multipath Rayleigh fading channel
5. AWGN noise addition
6. OFDM receiver (CP removal and FFT)
7. Channel estimation
8. Zero-Forcing (ZF) equalization
9. QPSK demodulation and bit recovery

## OFDM System Parameters

| Parameter | Value |
|----------|------|
| FFT size | 64 |
| Cyclic Prefix Length | 16 |
| Pilot Subcarriers | 8 |
| Data Subcarriers | 48 |
| Guard Subcarriers | 8 |
| Modulation | QPSK |
| Channel Model | Rayleigh Fading |
| Noise Model | AWGN |
| SNR Range | 0–30 dB |

## Channel Estimation Methods

### Least Squares (LS)
The LS estimator computes the channel estimate at pilot subcarriers by

Ĥ = Y / X

where Y is the received pilot symbol and X is the transmitted pilot symbol.  
Although computationally simple, LS amplifies noise and performs poorly at low SNR.

### Minimum Mean Square Error (MMSE)
MMSE improves LS by incorporating channel statistics and noise variance.  
It achieves lower estimation error but requires prior knowledge of channel correlation and SNR.

### CNN-Based Channel Estimator
The CNN model processes the LS estimate using a residual encoder-decoder architecture.

Architecture characteristics:
- 1D convolutional layers
- Batch normalization and ReLU activations
- Residual learning to refine LS estimate
- Approximately 270K trainable parameters

The CNN learns to remove noise from the LS estimate while preserving channel structure.

### BiLSTM-Based Channel Estimator
The BiLSTM treats the channel frequency response as a sequence across subcarriers.

Key properties:
- Three stacked bidirectional LSTM layers
- Hidden size of 128 units
- Fully connected regression head
- Approximately 1.02M parameters

Bidirectional processing allows the network to capture long-range correlations across subcarriers.

## Dataset Generation

Training data is generated using Monte Carlo simulation.

Dataset generation steps:
1. Random SNR is sampled between 0 and 30 dB
2. A new multipath Rayleigh channel is generated
3. Random QPSK symbols are transmitted
4. LS channel estimate is computed
5. True channel response is stored as the ground truth

Dataset statistics:

| Dataset | Samples |
|-------|--------|
| Training Set | 50,000 |
| Test Set | 5,000 |

Each sample consists of

Input: LS channel estimate  
Target: True channel frequency response

## Training Configuration

| Parameter | Value |
|----------|------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 256 |
| Training Epochs | 60 |
| LR Schedule | Cosine Annealing |

## Performance Evaluation

The estimators are evaluated using:

- Mean Square Error (MSE)
- Bit Error Rate (BER)

### Main Observations

- CNN reduces MSE by approximately **60% compared to LS** at 20 dB SNR
- CNN reduces MSE by approximately **35% compared to MMSE**
- BiLSTM slightly outperforms CNN at high SNR
- Deep learning estimators remain robust even with fewer pilot symbols
- Optimal cyclic prefix length must be at least equal to the maximum channel delay

### BER Improvement

Deep learning estimators achieve approximately **2 dB SNR gain** over MMSE at BER = 10⁻².

## Experimental Environment

Experiments were conducted using:

- Python 3.10
- PyTorch 2.x
- NVIDIA T4 GPU
- Google Colaboratory

Training time:
- CNN: ~12 minutes
- BiLSTM: ~25 minutes

## Results Summary

| SNR (dB) | LS MSE | MMSE MSE | CNN MSE | BiLSTM MSE |
|---------|--------|----------|---------|------------|
| 10 | 6.21e-2 | 2.93e-2 | 1.24e-2 | 9.33e-3 |
| 15 | 3.57e-2 | 1.65e-2 | 6.73e-3 | 5.05e-3 |
| 20 | 2.06e-2 | 9.30e-3 | 3.65e-3 | 2.74e-3 |

## Future Work

Potential research extensions include:

- Time-varying channel estimation with Doppler effects
- Transformer-based channel estimation architectures
- Online meta-learning for adaptive wireless environments
- Model compression for embedded 5G deployment
- Extension to massive MIMO-OFDM systems

## References

Key references for this work include:

- Dahlman et al., *5G NR: The Next Generation Wireless Access Technology*
- Ye et al., Deep learning for channel estimation in OFDM systems
- Soltani et al., Deep learning-based channel estimation
- Li et al., ReEsNet residual network for OFDM channel estimation
- Goodfellow et al., *Deep Learning*
