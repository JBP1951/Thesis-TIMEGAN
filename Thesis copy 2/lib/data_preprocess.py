# data_preprocess.py

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import glob

# -------------------------------------------------------------
# (1) Min-Max normalization
# -------------------------------------------------------------

def MinMax_Scaler(data):
    """
    Apply Min-Max normalization (0–1) to the entire dataset.
    Similar to TimeGAN's preprocessing approach.
    
    Args:
        data (np.ndarray): shape [N, features]
    
    Returns:
        norm_data (np.ndarray): normalized data
        scaler (MinMaxScaler): fitted scaler for inverse-transform
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_data = scaler.fit_transform(data)
    return norm_data, scaler

# -------------------------------------------------------------
# (2) Inverse transform
# -------------------------------------------------------------

def denormalize_data(norm_data, scaler):
    """
    Convert normalized data back to physical scale (inverse transform).
    """
    data_real = scaler.inverse_transform(norm_data)
    print("✅ Data converted back to physical scale.")
    return data_real


# -------------------------------------------------------------
# (4) Load and preprocess real signals
# -------------------------------------------------------------

def real_data_loading(data, seq_len, step=1, max_sequences=None):
    """
    Slice normalized data into fixed-length overlapping windows.
    Supports:
        - step (window stride)
        - max_sequences (to avoid RAM explosion)
    """
    sequences = []
    N = len(data)

    # Sliding-window con step correcto
    for i in range(0, N - seq_len, step):
        seq = data[i:i+seq_len]
        sequences.append(seq)

        if max_sequences is not None and len(sequences) >= max_sequences:
            break

    # Shuffle para simular i.i.d.
    idx = np.random.permutation(len(sequences))
    sequences = [sequences[i] for i in idx]

    print(f"✅ Created {len(sequences)} sequences | seq_len={seq_len}, step={step}")
    
    return sequences







# -----------------------------------------------------------
# (Optional) Synthetic Sine Wave Generator for Testing
# -----------------------------------------------------------

def sine_data_generation(no, seq_len, dim):
    """
    Generate synthetic sine-wave dataset (for debugging or testing TimeGAN).
    This is NOT used for real signals — only to test your architecture.

    Args:
        no (int): number of samples
        seq_len (int): sequence length
        dim (int): number of features (signals)
    
    Returns:
        data (list of np.ndarray): synthetic sine-wave data
    """
    data = []
    for i in range(no):
        temp = []
        for k in range(dim):
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)
        temp = np.transpose(np.asarray(temp))
        temp = (temp + 1) * 0.5  # normalize to [0, 1]
        data.append(temp)
    return data
# -------------------------------------------------------------
# (5) Unified loader (similar to TimeGAN's load_data)
# -------------------------------------------------------------

def load_data(data_type, seq_len, file_list=None, step=1, max_sequences=None):
    """
    Load and preprocess data for TimeGAN.
    Performs:
        - concatenation
        - MinMax normalization (ONE TIME ONLY)
        - sliding-window with step
        - sequence limit (max_sequences)
    """
    if data_type == "sine":
        no, dim = 10000, 5
        return sine_data_generation(no, seq_len, dim)

    elif data_type == "mytests":

        # 1) Load + concatenate all .mat files
        data = []
        for f in file_list:
            mat = sio.loadmat(f)
            data.append(mat['data_all'])

        data_global = np.vstack(data)
        print(f"📌 Raw concatenated data: {data_global.shape}")

        # 2) Global Min-Max normalization
        norm_data, scaler = MinMax_Scaler(data_global)
        print("📌 Applied MinMax normalization")

        # 3) Sliding-window + step + limit sequences
        sequences = real_data_loading(
            norm_data,
            seq_len,
            step=step,
            max_sequences=max_sequences
        )

        return sequences, scaler



# -------------------------------------------------------------
# (6) Batch generator (for training phase)
# -------------------------------------------------------------
def batch_generator(data,time, batch_size):
    """
    Mini-batch generator (same as in TimeGAN).
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]
    return X_mb,T_mb