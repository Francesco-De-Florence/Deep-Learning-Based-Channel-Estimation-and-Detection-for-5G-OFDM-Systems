# 🛰️ Deep Learning-Based Channel Estimation for 5G OFDM

**Project Overview:**
This notebook implements a full pipeline for OFDM channel estimation using:
- ✅ OFDM Transmitter / Receiver
- ✅ Multipath Rayleigh Fading Channel
- ✅ Traditional Estimators: LS and MMSE
- ✅ Deep Learning Estimators: CNN and LSTM
- ✅ BER vs SNR and MSE vs SNR analysis

**Author:** Francesco De Florence  
**Framework:** PyTorch + NumPy  
**Runtime:** GPU recommended (T4 on Colab)

## 📦 Section 1: Install & Import Dependencies
"""

# Install required packages
!pip install torch torchvision matplotlib numpy scipy tqdm -q

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import toeplitz
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ Using device: {device}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
print('✅ All dependencies loaded successfully!')

"""## ⚙️ Section 2: OFDM System Parameters"""

# ============================================================
#  OFDM System Configuration (5G NR inspired)
# ============================================================

class OFDMConfig:
    """5G-inspired OFDM system parameters."""
    # Subcarrier & OFDM
    N_FFT       = 64       # FFT size (total subcarriers)
    N_pilot     = 8        # Number of pilot subcarriers
    N_data      = 48       # Data subcarriers
    N_guard     = 8        # Guard subcarriers (null)
    CP_len      = 16       # Cyclic prefix length
    N_sym       = 14       # OFDM symbols per frame (5G slot)

    # Modulation
    MOD_ORDER   = 4        # QPSK (4-QAM)
    BITS_PER_SYM = 2       # log2(MOD_ORDER)

    # Channel
    N_paths     = 6        # Number of multipath taps
    MAX_DELAY   = 16       # Max delay spread (samples)

    # Training
    SNR_TRAIN_DB = 20      # SNR for dataset generation
    N_SAMPLES   = 50000    # Training samples
    N_TEST      = 5000     # Test samples

    # Pilot positions (evenly spaced)
    PILOT_IDX   = np.array([7, 14, 21, 28, 35, 42, 49, 56])
    PILOT_VAL   = np.ones(8, dtype=complex)  # BPSK pilots

cfg = OFDMConfig()

print('=' * 50)
print('  OFDM System Configuration')
print('=' * 50)
print(f'  FFT Size          : {cfg.N_FFT}')
print(f'  Data Subcarriers  : {cfg.N_data}')
print(f'  Pilot Subcarriers : {cfg.N_pilot}')
print(f'  Cyclic Prefix     : {cfg.CP_len}')
print(f'  Modulation        : {cfg.MOD_ORDER}-QAM (QPSK)')
print(f'  Multipath Taps    : {cfg.N_paths}')
print(f'  Training Samples  : {cfg.N_SAMPLES}')
print('=' * 50)

"""## 📡 Section 3: OFDM Transmitter"""

# ============================================================
#  OFDM Transmitter
# ============================================================

def qpsk_modulate(bits):
    """QPSK modulation: maps pairs of bits to complex symbols."""
    bits = bits.reshape(-1, 2)
    lut = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    idx = bits[:, 0] * 2 + bits[:, 1]
    return lut[idx]


def qpsk_demodulate(symbols):
    """QPSK hard demodulation."""
    bits = np.zeros((len(symbols), 2), dtype=int)
    bits[:, 0] = (np.real(symbols) < 0).astype(int)
    bits[:, 1] = (np.imag(symbols) < 0).astype(int)
    return bits.flatten()


def ofdm_transmit(bits, cfg):
    """
    OFDM Transmitter pipeline:
      bits -> QPSK -> IFFT -> Add CP -> time-domain signal
    Returns:
      tx_signal : time-domain transmitted samples
      tx_freq   : frequency-domain OFDM symbol (before IFFT)
    """
    # 1) Modulate data bits
    data_sym = qpsk_modulate(bits)  # shape: (N_data,)

    # 2) Build frequency-domain OFDM symbol
    tx_freq = np.zeros(cfg.N_FFT, dtype=complex)

    # Insert pilots
    tx_freq[cfg.PILOT_IDX] = cfg.PILOT_VAL

    # Insert data on remaining subcarriers
    data_idx = np.setdiff1d(
        np.arange(1, cfg.N_FFT - cfg.N_guard // 2),
        cfg.PILOT_IDX
    )[:cfg.N_data]
    tx_freq[data_idx] = data_sym

    # 3) IFFT
    tx_time = np.fft.ifft(tx_freq, n=cfg.N_FFT)

    # 4) Add cyclic prefix
    tx_signal = np.concatenate([tx_time[-cfg.CP_len:], tx_time])

    return tx_signal, tx_freq, data_idx


# Quick sanity check
test_bits = np.random.randint(0, 2, cfg.N_data * cfg.BITS_PER_SYM)
tx_sig, tx_f, d_idx = ofdm_transmit(test_bits, cfg)
print(f'✅ OFDM Transmitter OK')
print(f'   Bits in       : {len(test_bits)}')
print(f'   TX signal len : {len(tx_sig)} (FFT={cfg.N_FFT} + CP={cfg.CP_len})')
print(f'   Data idx len  : {len(d_idx)}')

"""## 🌊 Section 4: Multipath Rayleigh Fading Channel"""

# ============================================================
#  Multipath Rayleigh Fading Channel
# ============================================================

def generate_multipath_channel(cfg):
    """
    Generate a random multipath Rayleigh fading channel.
    Returns:
      h_time : time-domain channel impulse response (complex)
      H_freq : frequency-domain channel (FFT of h_time)
    """
    # Random tap delays (sorted)
    delays = np.sort(
        np.random.choice(cfg.MAX_DELAY, cfg.N_paths, replace=False)
    )

    # Rayleigh fading taps (exponential power delay profile)
    power_profile = np.exp(-0.2 * np.arange(cfg.N_paths))
    power_profile /= power_profile.sum()  # normalize

    # Complex Gaussian taps
    h_taps = (
        np.random.randn(cfg.N_paths) + 1j * np.random.randn(cfg.N_paths)
    ) * np.sqrt(power_profile / 2)

    # Build CIR vector
    h_time = np.zeros(cfg.N_FFT, dtype=complex)
    for i, d in enumerate(delays):
        h_time[d] += h_taps[i]

    # Frequency-domain channel
    H_freq = np.fft.fft(h_time, n=cfg.N_FFT)

    return h_time, H_freq


def apply_channel(tx_signal, h_time, snr_db):
    """
    Pass signal through multipath channel + AWGN.
    Returns received time-domain signal.
    """
    # Convolve (linear convolution)
    h_nonzero = h_time[:np.where(h_time != 0)[0][-1] + 1] if np.any(h_time != 0) else h_time[:1]
    rx_conv = np.convolve(tx_signal, h_nonzero)
    rx_conv = rx_conv[:len(tx_signal)]  # truncate

    # Add AWGN
    sig_power = np.mean(np.abs(rx_conv) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*rx_conv.shape) + 1j * np.random.randn(*rx_conv.shape)
    )
    return rx_conv + noise


def ofdm_receive(rx_signal, cfg):
    """
    OFDM Receiver: Remove CP -> FFT -> frequency-domain symbol
    """
    # Remove cyclic prefix
    rx_no_cp = rx_signal[cfg.CP_len: cfg.CP_len + cfg.N_FFT]
    # FFT
    Y = np.fft.fft(rx_no_cp, n=cfg.N_FFT)
    return Y


# Visualize a sample channel
h_t, H_f = generate_multipath_channel(cfg)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Multipath Rayleigh Fading Channel', fontsize=14, fontweight='bold')

axes[0].stem(np.arange(cfg.N_FFT)[:30], np.abs(h_t[:30]),
             linefmt='C0-', markerfmt='C0o', basefmt='gray')
axes[0].set_title('Channel Impulse Response (CIR)')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('|h[n]|')
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.abs(H_f), color='C1', linewidth=1.5)
axes[1].set_title('Channel Frequency Response (CFR)')
axes[1].set_xlabel('Subcarrier Index')
axes[1].set_ylabel('|H[k]|')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('channel_response.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Channel model visualized.')

"""## 📐 Section 5: Traditional Channel Estimators (LS & MMSE)"""

# ============================================================
#  Traditional Channel Estimators
# ============================================================

def ls_estimator(Y, cfg):
    """
    Least Squares (LS) Channel Estimation.
    At pilot positions: H_ls[p] = Y[p] / X[p]
    Then interpolate to all subcarriers.
    """
    # LS estimate at pilots
    H_pilots = Y[cfg.PILOT_IDX] / cfg.PILOT_VAL

    # Linear interpolation over all subcarriers
    H_ls = np.interp(
        np.arange(cfg.N_FFT),
        cfg.PILOT_IDX,
        H_pilots.real
    ) + 1j * np.interp(
        np.arange(cfg.N_FFT),
        cfg.PILOT_IDX,
        H_pilots.imag
    )
    return H_ls


def mmse_estimator(Y, cfg, snr_db):
    """
    MMSE Channel Estimation.
    Uses correlation-based smoothing of LS estimate.
    MMSE = R_hh * (R_hh + sigma^2 * (X^H X)^-1)^-1 * H_ls_pilots
    """
    snr_lin = 10 ** (snr_db / 10)
    Np = len(cfg.PILOT_IDX)

    # LS at pilots
    H_ls_p = Y[cfg.PILOT_IDX] / cfg.PILOT_VAL

    # Build channel correlation matrix (exponential model)
    # R_hh[i,j] = exp(-|i-j| * alpha) for pilot grid
    alpha = 0.5
    p_idx = np.arange(Np)
    R_hh = np.exp(-alpha * np.abs(p_idx[:, None] - p_idx[None, :]))

    # Noise regularization: beta = (1/SNR) * trace(X^H X) / N
    beta = 1.0 / snr_lin
    W = R_hh @ np.linalg.inv(R_hh + beta * np.eye(Np))

    # MMSE estimate at pilots
    H_mmse_p = W @ H_ls_p

    # Interpolate to full band
    H_mmse = np.interp(
        np.arange(cfg.N_FFT),
        cfg.PILOT_IDX,
        H_mmse_p.real
    ) + 1j * np.interp(
        np.arange(cfg.N_FFT),
        cfg.PILOT_IDX,
        H_mmse_p.imag
    )
    return H_mmse


# Test estimators on one OFDM symbol
bits = np.random.randint(0, 2, cfg.N_data * cfg.BITS_PER_SYM)
tx_sig, tx_f, d_idx = ofdm_transmit(bits, cfg)
h_t, H_true = generate_multipath_channel(cfg)
rx_sig = apply_channel(tx_sig, h_t, snr_db=20)
Y = ofdm_receive(rx_sig, cfg)

H_ls   = ls_estimator(Y, cfg)
H_mmse = mmse_estimator(Y, cfg, snr_db=20)

mse_ls   = np.mean(np.abs(H_ls   - H_true) ** 2)
mse_mmse = np.mean(np.abs(H_mmse - H_true) ** 2)

print(f'✅ Traditional Estimators (SNR=20dB):')
print(f'   MSE (LS)   = {mse_ls:.6f}')
print(f'   MSE (MMSE) = {mse_mmse:.6f}')

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
k = np.arange(cfg.N_FFT)
ax.plot(k, np.abs(H_true), 'k-',  lw=2,   label='True Channel')
ax.plot(k, np.abs(H_ls),   'r--', lw=1.5, label=f'LS  (MSE={mse_ls:.4f})')
ax.plot(k, np.abs(H_mmse), 'b-.',  lw=1.5, label=f'MMSE (MSE={mse_mmse:.4f})')
ax.scatter(cfg.PILOT_IDX, np.abs(H_true[cfg.PILOT_IDX]),
           c='green', zorder=5, s=60, label='Pilot positions')
ax.set_xlabel('Subcarrier Index')
ax.set_ylabel('|H[k]|')
ax.set_title('Channel Estimation: LS vs MMSE (SNR=20dB)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ls_mmse_estimation.png', dpi=150, bbox_inches='tight')
plt.show()

"""## 🧠 Section 6: Deep Learning Models (CNN & LSTM)"""

# ============================================================
#  Deep Learning Channel Estimators
# ============================================================

# ---- CNN Estimator ----------------------------------------
class CNNChannelEstimator(nn.Module):
    """
    CNN-based channel estimator.
    Input : LS estimate [real, imag] -> (batch, 2, N_FFT)
    Output: True channel [real, imag] -> (batch, 2, N_FFT)
    """
    def __init__(self, n_fft=64):
        super(CNNChannelEstimator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=9, padding=4),
        )
        # Residual connection
        self.residual = nn.Conv1d(2, 2, kernel_size=1)

    def forward(self, x):
        res = self.residual(x)
        out = self.encoder(x)
        out = self.decoder(out)
        return out + res  # residual learning


# ---- LSTM Estimator ---------------------------------------
class LSTMChannelEstimator(nn.Module):
    """
    Bidirectional LSTM channel estimator.
    Input : LS estimate sequence (batch, N_FFT, 2)
    Output: Estimated channel     (batch, N_FFT, 2)
    """
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, n_fft=64):
        super(LSTMChannelEstimator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        # x: (batch, 2, N_FFT) -> (batch, N_FFT, 2)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out)  # (batch, N_FFT, 2)
        return out.permute(0, 2, 1)  # back to (batch, 2, N_FFT)


# Instantiate models
cnn_model  = CNNChannelEstimator(n_fft=cfg.N_FFT).to(device)
lstm_model = LSTMChannelEstimator(n_fft=cfg.N_FFT).to(device)

# Parameter count
cnn_params  = sum(p.numel() for p in cnn_model.parameters())
lstm_params = sum(p.numel() for p in lstm_model.parameters())

print(f'✅ CNN  Model: {cnn_params:,} parameters')
print(f'✅ LSTM Model: {lstm_params:,} parameters')

# Quick forward pass test
dummy = torch.randn(4, 2, cfg.N_FFT).to(device)
print(f'   CNN  output shape : {cnn_model(dummy).shape}')
print(f'   LSTM output shape : {lstm_model(dummy).shape}')

"""## 🗃️ Section 7: Dataset Generation"""

# ============================================================
#  Dataset Generation
# ============================================================

def generate_dataset(n_samples, cfg, snr_db_range=(0, 30)):
    """
    Generate dataset of (LS_estimate, true_channel) pairs.
    Each sample uses a different random channel realization.

    Returns:
      X : input  (n_samples, 2, N_FFT)  - LS estimate [re, im]
      Y : target (n_samples, 2, N_FFT)  - True channel [re, im]
    """
    X = np.zeros((n_samples, 2, cfg.N_FFT), dtype=np.float32)
    Y = np.zeros((n_samples, 2, cfg.N_FFT), dtype=np.float32)

    bits_per_frame = cfg.N_data * cfg.BITS_PER_SYM

    for i in tqdm(range(n_samples), desc='Generating dataset', ncols=80):
        # Random SNR from range
        snr_db = np.random.uniform(*snr_db_range)

        # Random channel
        h_t, H_true = generate_multipath_channel(cfg)

        # Transmit
        bits = np.random.randint(0, 2, bits_per_frame)
        tx_sig, _, _ = ofdm_transmit(bits, cfg)

        # Channel + noise
        rx_sig = apply_channel(tx_sig, h_t, snr_db=snr_db)

        # Receive
        Y_rx = ofdm_receive(rx_sig, cfg)

        # LS estimate as input feature
        H_ls = ls_estimator(Y_rx, cfg)

        X[i, 0, :] = H_ls.real
        X[i, 1, :] = H_ls.imag
        Y[i, 0, :] = H_true.real
        Y[i, 1, :] = H_true.imag

    return X, Y


print('📊 Generating training dataset...')
X_train, Y_train = generate_dataset(cfg.N_SAMPLES, cfg, snr_db_range=(0, 30))
print('📊 Generating test dataset...')
X_test,  Y_test  = generate_dataset(cfg.N_TEST,    cfg, snr_db_range=(0, 30))

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
Y_train_t = torch.FloatTensor(Y_train)
X_test_t  = torch.FloatTensor(X_test)
Y_test_t  = torch.FloatTensor(Y_test)

train_loader = DataLoader(
    TensorDataset(X_train_t, Y_train_t),
    batch_size=256, shuffle=True, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    TensorDataset(X_test_t, Y_test_t),
    batch_size=256, shuffle=False
)

print(f'\n✅ Dataset ready:')
print(f'   Train: X={X_train.shape}, Y={Y_train.shape}')
print(f'   Test : X={X_test.shape},  Y={Y_test.shape}')

"""## 🏋️ Section 8: Training Loop"""

# ============================================================
#  Training Function
# ============================================================

def train_model(model, train_loader, test_loader, n_epochs=50,
                lr=1e-3, model_name='Model'):
    """
    Train a deep learning channel estimator.
    Returns training history.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')

    print(f'\n🏋️  Training {model_name} for {n_epochs} epochs...')
    print('-' * 60)

    for epoch in range(1, n_epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(test_loader.dataset)

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'best_{model_name.lower()}.pt')

        if epoch % 10 == 0 or epoch == 1:
            print(f'  Epoch [{epoch:3d}/{n_epochs}]  '
                  f'Train Loss: {train_loss:.6f}  '
                  f'Val Loss: {val_loss:.6f}  '
                  f'LR: {scheduler.get_last_lr()[0]:.2e}')

    # Load best weights
    model.load_state_dict(torch.load(f'best_{model_name.lower()}.pt'))
    print(f'  ✅ Best Val Loss: {best_val:.6f}')
    return history


# Train CNN
cnn_history  = train_model(cnn_model,  train_loader, test_loader,
                            n_epochs=25, lr=1e-3, model_name='CNN')

# Train LSTM
lstm_history = train_model(lstm_model, train_loader, test_loader,
                            n_epochs=25, lr=1e-3, model_name='LSTM')

"""## 📉 Section 9: Training Curves"""

# ============================================================
#  Plot Training Curves
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History', fontsize=14, fontweight='bold')

for ax, hist, name, color in zip(
    axes,
    [cnn_history, lstm_history],
    ['CNN', 'LSTM'],
    ['royalblue', 'darkorange']
):
    epochs = range(1, len(hist['train_loss']) + 1)
    ax.semilogy(epochs, hist['train_loss'], color=color,
                lw=2, label='Train Loss')
    ax.semilogy(epochs, hist['val_loss'], color=color,
                lw=2, ls='--', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.set_title(f'{name} Training Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Training curves plotted.')

"""## 📊 Section 10: MSE vs SNR Analysis"""

# ============================================================
#  MSE vs SNR for all estimators
# ============================================================

def evaluate_mse_snr(cfg, snr_range, n_trials=500):
    """
    Evaluate MSE for LS, MMSE, CNN, LSTM at each SNR.
    """
    results = {
        'LS': [], 'MMSE': [], 'CNN': [], 'LSTM': []
    }

    cnn_model.eval()
    lstm_model.eval()

    bits_per_frame = cfg.N_data * cfg.BITS_PER_SYM

    for snr_db in tqdm(snr_range, desc='MSE vs SNR', ncols=80):
        mse = {'LS': 0, 'MMSE': 0, 'CNN': 0, 'LSTM': 0}

        # Batch DL inference
        X_batch = []
        H_true_batch = []

        for _ in range(n_trials):
            h_t, H_true = generate_multipath_channel(cfg)
            bits = np.random.randint(0, 2, bits_per_frame)
            tx_sig, _, _ = ofdm_transmit(bits, cfg)
            rx_sig = apply_channel(tx_sig, h_t, snr_db=snr_db)
            Y_rx = ofdm_receive(rx_sig, cfg)

            H_ls   = ls_estimator(Y_rx, cfg)
            H_mmse = mmse_estimator(Y_rx, cfg, snr_db=snr_db)

            mse['LS']   += np.mean(np.abs(H_ls   - H_true) ** 2)
            mse['MMSE'] += np.mean(np.abs(H_mmse - H_true) ** 2)

            X_batch.append([H_ls.real, H_ls.imag])
            H_true_batch.append([H_true.real, H_true.imag])

        # DL batch inference
        X_t = torch.FloatTensor(np.array(X_batch)).to(device)
        H_t = np.array(H_true_batch)  # (n_trials, 2, N_FFT)

        with torch.no_grad():
            cnn_pred  = cnn_model(X_t).cpu().numpy()
            lstm_pred = lstm_model(X_t).cpu().numpy()

        # Reconstruct complex
        H_cnn_c  = cnn_pred[:,  0, :] + 1j * cnn_pred[:,  1, :]
        H_lstm_c = lstm_pred[:, 0, :] + 1j * lstm_pred[:, 1, :]
        H_true_c = H_t[:, 0, :] + 1j * H_t[:, 1, :]

        mse['CNN']  = np.mean(np.abs(H_cnn_c  - H_true_c) ** 2)
        mse['LSTM'] = np.mean(np.abs(H_lstm_c - H_true_c) ** 2)

        for k in ['LS', 'MMSE']:
            results[k].append(mse[k] / n_trials)
        for k in ['CNN', 'LSTM']:
            results[k].append(mse[k])

    return results


snr_range = np.arange(0, 31, 5)  # 0 to 30 dB, step 5
print('📊 Evaluating MSE vs SNR...')
mse_results = evaluate_mse_snr(cfg, snr_range, n_trials=300)

# Plot MSE vs SNR
plt.figure(figsize=(10, 6))
styles = {
    'LS':   ('red',       's', '--'),
    'MMSE': ('royalblue', '^', '-.'),
    'CNN':  ('green',     'o', '-'),
    'LSTM': ('darkorange','D', '-'),
}
for name, (color, marker, ls) in styles.items():
    plt.semilogy(snr_range, mse_results[name],
                 color=color, marker=marker, ls=ls,
                 lw=2, markersize=8, label=name)

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('MSE', fontsize=13)
plt.title('Channel Estimation: MSE vs SNR', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('mse_vs_snr.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ MSE vs SNR plotted.')

"""## 📶 Section 11: BER vs SNR Analysis"""

# ============================================================
#  BER vs SNR for all estimators
# ============================================================

def equalize_and_decode(Y_rx, H_est, cfg, data_idx):
    """
    One-tap equalization (ZF) + QPSK demodulation.
    """
    # Equalize
    Y_eq = Y_rx / (H_est + 1e-10)  # ZF equalizer
    # Extract data subcarriers
    data_sym_rx = Y_eq[data_idx]
    # Demodulate
    bits_rx = qpsk_demodulate(data_sym_rx)
    return bits_rx


def evaluate_ber_snr(cfg, snr_range, n_trials=200):
    """
    Evaluate BER for LS, MMSE, CNN, LSTM + perfect CSI at each SNR.
    """
    results = {'LS': [], 'MMSE': [], 'CNN': [], 'LSTM': [], 'Perfect CSI': []}

    cnn_model.eval()
    lstm_model.eval()
    bits_per_frame = cfg.N_data * cfg.BITS_PER_SYM

    for snr_db in tqdm(snr_range, desc='BER vs SNR', ncols=80):
        errors = {k: 0 for k in results}
        total_bits = 0

        X_batch, bits_list = [], []
        h_true_list, Y_rx_list, d_idx_list = [], [], []

        for _ in range(n_trials):
            bits = np.random.randint(0, 2, bits_per_frame)
            h_t, H_true = generate_multipath_channel(cfg)
            tx_sig, _, d_idx = ofdm_transmit(bits, cfg)
            rx_sig = apply_channel(tx_sig, h_t, snr_db=snr_db)
            Y_rx = ofdm_receive(rx_sig, cfg)

            H_ls   = ls_estimator(Y_rx, cfg)
            H_mmse = mmse_estimator(Y_rx, cfg, snr_db=snr_db)

            # LS
            b_rx = equalize_and_decode(Y_rx, H_ls, cfg, d_idx)
            errors['LS'] += np.sum(bits != b_rx[:len(bits)])

            # MMSE
            b_rx = equalize_and_decode(Y_rx, H_mmse, cfg, d_idx)
            errors['MMSE'] += np.sum(bits != b_rx[:len(bits)])

            # Perfect CSI
            b_rx = equalize_and_decode(Y_rx, H_true, cfg, d_idx)
            errors['Perfect CSI'] += np.sum(bits != b_rx[:len(bits)])

            total_bits += len(bits)
            X_batch.append([H_ls.real, H_ls.imag])
            bits_list.append(bits)
            h_true_list.append(H_true)
            Y_rx_list.append(Y_rx)
            d_idx_list.append(d_idx)

        # DL batch
        X_t = torch.FloatTensor(np.array(X_batch)).to(device)
        with torch.no_grad():
            cnn_out  = cnn_model(X_t).cpu().numpy()
            lstm_out = lstm_model(X_t).cpu().numpy()

        for i in range(n_trials):
            H_cnn_i  = cnn_out[i,  0, :] + 1j * cnn_out[i,  1, :]
            H_lstm_i = lstm_out[i, 0, :] + 1j * lstm_out[i, 1, :]

            b_rx = equalize_and_decode(Y_rx_list[i], H_cnn_i,  cfg, d_idx_list[i])
            errors['CNN']  += np.sum(bits_list[i] != b_rx[:len(bits_list[i])])

            b_rx = equalize_and_decode(Y_rx_list[i], H_lstm_i, cfg, d_idx_list[i])
            errors['LSTM'] += np.sum(bits_list[i] != b_rx[:len(bits_list[i])])

        for k in results:
            ber = max(errors[k] / total_bits, 1e-6)  # floor for log plot
            results[k].append(ber)

    return results


snr_ber_range = np.arange(0, 31, 3)
print('📊 Evaluating BER vs SNR...')
ber_results = evaluate_ber_snr(cfg, snr_ber_range, n_trials=150)

# Plot BER vs SNR
plt.figure(figsize=(10, 6))
styles_ber = {
    'LS':          ('red',       's', '--'),
    'MMSE':        ('royalblue', '^', '-.'),
    'CNN':         ('green',     'o', '-'),
    'LSTM':        ('darkorange','D', '-'),
    'Perfect CSI': ('black',     'x', ':'),
}
for name, (color, marker, ls) in styles_ber.items():
    plt.semilogy(snr_ber_range, ber_results[name],
                 color=color, marker=marker, ls=ls,
                 lw=2, markersize=8, label=name)

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('Bit Error Rate (BER)', fontsize=13)
plt.title('OFDM BER vs SNR — All Estimators', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, which='both', alpha=0.3)
plt.ylim([1e-4, 1])
plt.tight_layout()
plt.savefig('ber_vs_snr.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ BER vs SNR plotted.')

"""## 🔬 Section 12: Visual Channel Estimation Comparison"""

# ============================================================
#  Visual Comparison: all estimators on one OFDM symbol
# ============================================================

test_snr = 15  # dB

bits = np.random.randint(0, 2, cfg.N_data * cfg.BITS_PER_SYM)
h_t, H_true = generate_multipath_channel(cfg)
tx_sig, _, d_idx = ofdm_transmit(bits, cfg)
rx_sig = apply_channel(tx_sig, h_t, snr_db=test_snr)
Y_rx = ofdm_receive(rx_sig, cfg)

H_ls_v   = ls_estimator(Y_rx, cfg)
H_mmse_v = mmse_estimator(Y_rx, cfg, snr_db=test_snr)

# DL estimations
X_in = torch.FloatTensor([[H_ls_v.real, H_ls_v.imag]]).to(device)
with torch.no_grad():
    H_cnn_v  = cnn_model(X_in).cpu().numpy()[0]
    H_lstm_v = lstm_model(X_in).cpu().numpy()[0]

H_cnn_v  = H_cnn_v[0]  + 1j * H_cnn_v[1]
H_lstm_v = H_lstm_v[0] + 1j * H_lstm_v[1]

k = np.arange(cfg.N_FFT)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f'Channel Estimation Comparison (SNR={test_snr}dB)',
             fontsize=14, fontweight='bold')

estimators = [
    ('LS',   H_ls_v,   'red'),
    ('MMSE', H_mmse_v, 'royalblue'),
    ('CNN',  H_cnn_v,  'green'),
    ('LSTM', H_lstm_v, 'darkorange'),
]

for ax, (name, H_est, color) in zip(axes.flat, estimators):
    mse_val = np.mean(np.abs(H_est - H_true) ** 2)
    ax.plot(k, np.abs(H_true), 'k-',  lw=2.5, label='True Channel', zorder=3)
    ax.plot(k, np.abs(H_est),  color=color, lw=1.8, ls='--',
            label=f'{name} (MSE={mse_val:.5f})')
    ax.scatter(cfg.PILOT_IDX, np.abs(H_true[cfg.PILOT_IDX]),
               c='lime', zorder=5, s=50, label='Pilots')
    ax.set_title(f'{name} Estimator', fontweight='bold')
    ax.set_xlabel('Subcarrier')
    ax.set_ylabel('|H[k]|')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visual_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Visual comparison done.')

"""## 📋 Section 13: Summary Table & Final Report"""

# ============================================================
#  Summary Table
# ============================================================

import pandas as pd

snr_points = [0, 5, 10, 15, 20, 25, 30]
snr_idx = [list(snr_range).index(s) for s in snr_points if s in list(snr_range)]
snr_ber_idx = [list(snr_ber_range).index(s) for s in snr_points if s in list(snr_ber_range)]

# MSE Table
mse_table = pd.DataFrame({
    'SNR (dB)':    [snr_range[i]   for i in snr_idx],
    'LS MSE':      [f"{mse_results['LS'][i]:.6f}"   for i in snr_idx],
    'MMSE MSE':    [f"{mse_results['MMSE'][i]:.6f}"  for i in snr_idx],
    'CNN MSE':     [f"{mse_results['CNN'][i]:.6f}"   for i in snr_idx],
    'LSTM MSE':    [f"{mse_results['LSTM'][i]:.6f}"  for i in snr_idx],
})

# BER Table
ber_table = pd.DataFrame({
    'SNR (dB)':       [snr_ber_range[i]       for i in snr_ber_idx],
    'LS BER':         [f"{ber_results['LS'][i]:.4f}"          for i in snr_ber_idx],
    'MMSE BER':       [f"{ber_results['MMSE'][i]:.4f}"         for i in snr_ber_idx],
    'CNN BER':        [f"{ber_results['CNN'][i]:.4f}"          for i in snr_ber_idx],
    'LSTM BER':       [f"{ber_results['LSTM'][i]:.4f}"         for i in snr_ber_idx],
    'Perfect CSI':    [f"{ber_results['Perfect CSI'][i]:.4f}"  for i in snr_ber_idx],
})

print('=' * 70)
print('  MSE vs SNR Summary')
print('=' * 70)
print(mse_table.to_string(index=False))

print('\n')
print('=' * 80)
print('  BER vs SNR Summary')
print('=' * 80)
print(ber_table.to_string(index=False))

# Model sizes
print('\n')
print('=' * 40)
print('  Model Parameters')
print('=' * 40)
print(f'  CNN  : {cnn_params:,} params')
print(f'  LSTM : {lstm_params:,} params')
print('=' * 40)

"""## 🎨 Section 14: Combined Final Dashboard"""

# ============================================================
#  Final Dashboard: all key plots in one figure
# ============================================================

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

fig.suptitle(
    'Deep Learning-Based Channel Estimation for 5G OFDM\nFull Performance Dashboard',
    fontsize=15, fontweight='bold', y=1.01
)

colors = {'LS': 'red', 'MMSE': 'royalblue', 'CNN': 'green',
          'LSTM': 'darkorange', 'Perfect CSI': 'black'}
markers = {'LS': 's', 'MMSE': '^', 'CNN': 'o', 'LSTM': 'D', 'Perfect CSI': 'x'}

# --- (0,0) MSE vs SNR ---
ax1 = fig.add_subplot(gs[0, 0])
for name in ['LS', 'MMSE', 'CNN', 'LSTM']:
    ax1.semilogy(snr_range, mse_results[name],
                 color=colors[name], marker=markers[name],
                 lw=2, markersize=6, label=name)
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('MSE')
ax1.set_title('MSE vs SNR', fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, which='both', alpha=0.3)

# --- (0,1) BER vs SNR ---
ax2 = fig.add_subplot(gs[0, 1])
for name in ['LS', 'MMSE', 'CNN', 'LSTM', 'Perfect CSI']:
    ax2.semilogy(snr_ber_range, ber_results[name],
                 color=colors[name], marker=markers[name],
                 lw=2, markersize=6, label=name)
ax2.set_xlabel('SNR (dB)')
ax2.set_ylabel('BER')
ax2.set_title('BER vs SNR', fontweight='bold')
ax2.legend(fontsize=7)
ax2.grid(True, which='both', alpha=0.3)
ax2.set_ylim([1e-4, 1])

# --- (0,2) Training curves ---
ax3 = fig.add_subplot(gs[0, 2])
e_c = range(1, len(cnn_history['val_loss']) + 1)
e_l = range(1, len(lstm_history['val_loss']) + 1)
ax3.semilogy(e_c, cnn_history['val_loss'],  color='green',      lw=2, label='CNN Val')
ax3.semilogy(e_l, lstm_history['val_loss'], color='darkorange',  lw=2, label='LSTM Val')
ax3.semilogy(e_c, cnn_history['train_loss'],  color='green',    lw=1.2, ls='--', label='CNN Train')
ax3.semilogy(e_l, lstm_history['train_loss'], color='darkorange', lw=1.2, ls='--', label='LSTM Train')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Training Curves', fontweight='bold')
ax3.legend(fontsize=7)
ax3.grid(True, which='both', alpha=0.3)

# --- (1,0) Channel CIR ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.stem(np.arange(25), np.abs(h_t[:25]),
         linefmt='C0-', markerfmt='C0o', basefmt='gray')
ax4.set_title('Sample CIR', fontweight='bold')
ax4.set_xlabel('Tap')
ax4.set_ylabel('|h[n]|')
ax4.grid(True, alpha=0.3)

# --- (1,1) Channel CFR ---
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(k, np.abs(H_true),   'k-',  lw=2.5, label='True')
ax5.plot(k, np.abs(H_ls_v),   'r--', lw=1.5, label='LS')
ax5.plot(k, np.abs(H_mmse_v), 'b-.', lw=1.5, label='MMSE')
ax5.plot(k, np.abs(H_cnn_v),  color='green',      lw=1.5, ls='-', label='CNN')
ax5.plot(k, np.abs(H_lstm_v), color='darkorange',  lw=1.5, ls='-', label='LSTM')
ax5.scatter(cfg.PILOT_IDX, np.abs(H_true[cfg.PILOT_IDX]),
            c='lime', zorder=5, s=40)
ax5.set_title(f'CFR Estimation (SNR={test_snr}dB)', fontweight='bold')
ax5.set_xlabel('Subcarrier')
ax5.set_ylabel('|H[k]|')
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)

# --- (1,2) MSE improvement bar chart at 20dB ---
ax6 = fig.add_subplot(gs[1, 2])
idx_20 = list(snr_range).index(20) if 20 in snr_range else -1
if idx_20 >= 0:
    names  = ['LS', 'MMSE', 'CNN', 'LSTM']
    vals   = [mse_results[n][idx_20] for n in names]
    bar_c  = [colors[n] for n in names]
    bars = ax6.bar(names, vals, color=bar_c, edgecolor='white', linewidth=1.5)
    ax6.set_yscale('log')
    ax6.set_title('MSE at SNR=20dB', fontweight='bold')
    ax6.set_ylabel('MSE (log)')
    for bar, val in zip(bars, vals):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 val * 1.2, f'{val:.4f}',
                 ha='center', va='bottom', fontsize=8)
    ax6.grid(True, axis='y', alpha=0.3)

plt.savefig('final_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Final dashboard saved as final_dashboard.png')

"""## 💾 Section 15: Save Models & Results"""

# ============================================================
#  Save everything
# ============================================================

import json

# Save models
torch.save(cnn_model.state_dict(),  'cnn_channel_estimator.pt')
torch.save(lstm_model.state_dict(), 'lstm_channel_estimator.pt')

# Save results
results_dict = {
    'snr_mse_db':  snr_range.tolist(),
    'snr_ber_db':  snr_ber_range.tolist(),
    'mse_results': {k: [float(v) for v in vals] for k, vals in mse_results.items()},
    'ber_results': {k: [float(v) for v in vals] for k, vals in ber_results.items()},
    'cnn_train_loss':  cnn_history['train_loss'],
    'lstm_train_loss': lstm_history['train_loss'],
    'cnn_val_loss':    cnn_history['val_loss'],
    'lstm_val_loss':   lstm_history['val_loss'],
}

with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print('✅ Models saved:')
print('   cnn_channel_estimator.pt')
print('   lstm_channel_estimator.pt')
print('✅ Results saved: results.json')
print('✅ Figures saved: channel_response.png, mse_vs_snr.png,')
print('                  ber_vs_snr.png, final_dashboard.png')
print()
print('=' * 55)
print('  🎉 Project Complete!')
print('=' * 55)
print('  Summary:')
print('  - OFDM transmitter/receiver with CP implemented')
print('  - Multipath Rayleigh fading channel modeled')
print('  - LS + MMSE traditional estimators evaluated')
print('  - CNN + LSTM deep learning estimators trained')
print('  - BER and MSE vs SNR curves generated')
print('=' * 55)
