# data_preprocess.py

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import glob
import h5py

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
    Carga archivos MATLAB v7.3 (HDF5) con señales procesadas.
    Devuelve:
        - sequences: ventanas de tamaño seq_len
        - all_scalers: lista de normalizadores (uno por archivo)
        - feature_names: nombres de columnas
    """

    if data_type != "mytests":
        raise NotImplementedError("Only mytests is implemented.")

    mat_path = file_list[0]
    print(f"📌 Loading: {mat_path}")

    sequences = []       # todas las ventanas se juntan aquí
    all_scalers = []     # scaler por archivo
    feature_names = None # se llena solo una vez

    with h5py.File(mat_path, "r") as f:

        data_all = f["data_all"]
        n_files  = data_all.shape[1]

        for i in range(n_files):

            ref = data_all[0][i]
            entry = f[ref]

            sig_group = entry["signals_processed"]

            # === 1) Orden estable de columnas ===
            sig_names = sorted(list(sig_group.keys()))

            # Guardar feature names solo una vez
            if feature_names is None:
                feature_names = sig_names
                print(f"📌 Feature order: {feature_names}")

            # === 2) Leer señales ===
            signals = []
            for s in sig_names:
                vec = np.array(sig_group[s][:]).reshape(-1)
                signals.append(vec)

            # formato final:  N × features
            file_data = np.vstack(signals).T

            # === 3) Normalización por archivo ===
            scaler = MinMaxScaler()
            file_data = scaler.fit_transform(file_data)
            all_scalers.append(scaler)

            # === 4) Crear ventanas ===
            N = len(file_data)
            for j in range(0, N - seq_len, step):

                seq = file_data[j:j + seq_len]
                sequences.append(seq)

                if max_sequences and len(sequences) >= max_sequences:
                    break

            if max_sequences and len(sequences) >= max_sequences:
                break

    # === Conversión a array ===
    sequences = np.asarray(sequences, dtype=np.float32)

    print(f"📌 Total sequences: {len(sequences)} | Each: {seq_len}×{sequences.shape[2]}")

    return sequences, all_scalers, feature_names





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

    X_mb = np.asarray([data[i] for i in train_idx], dtype=np.float32)

    T_mb = np.asarray([time[i] for i in train_idx], dtype=np.int32)

    return X_mb,T_mb