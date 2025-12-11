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

    if data_type != "mytests":
        raise NotImplementedError("Only mytests is implemented.")

    mat_path = file_list[0]
    print(f"📌 Loading: {mat_path}")

    # Listas de salida
    X_list        = []   # acelerómetros
    C_time_list   = []   # rpm, temp, current
    C_static_list = []   # peso, distancia
    all_scalers   = []   # un scaler por experimento
    feature_names = None

    # Nombres de señales
    accel_names     = ['Accel1', 'Accel2', 'Accel3', 'Accel4']
    c_time_names    = ['Motor_current', 'Speed', 'Temperature']
    all_signal_names = accel_names + c_time_names

    import h5py
    with h5py.File(mat_path, "r") as f:

        data_all = f["data_all"]
        n_files  = data_all.shape[1]

        for i in range(n_files):

            ref   = data_all[0][i]
            entry = f[ref]

            sig_group = entry["signals_processed"]
            meta      = entry["meta"]

            # Guardar feature names solo una vez
            if feature_names is None:
                feature_names = all_signal_names
                print(f"📌 Feature order: {feature_names}")

            # -------- 1) LEER SEÑALES DINÁMICAS --------
            # Acelerómetros
            accel_signals = []
            for name in accel_names:
                vec = np.array(sig_group[name][:]).reshape(-1)
                accel_signals.append(vec)
            accel_data = np.vstack(accel_signals).T    # [N, 4]

            # Condicionales en el tiempo
            c_time_signals = []
            for name in c_time_names:
                vec = np.array(sig_group[name][:]).reshape(-1)
                c_time_signals.append(vec)
            c_time_data = np.vstack(c_time_signals).T  # [N, 3]

            # Comprobar misma longitud
            N = accel_data.shape[0]
            if c_time_data.shape[0] != N:
                raise ValueError(f"Longitudes distintas en señales dinámicas en experimento {i}")

            # -------- 2) NORMALIZACIÓN CONJUNTA --------
            # -------- 2) NORMALIZACIÓN CONJUNTA (X + C_time + C_static) --------

            # 1) Repetimos C_static para que tenga longitud N (igual que X y C_time)
            c_static_repeat = np.repeat(
                np.array(c_static_vec).reshape(1, -1),
                accel_data.shape[0],
                axis=0
            )  # → [N, 2]

            # 2) Concatenamos TODO para normalizar junto
            full_data = np.concatenate(
                [accel_data, c_time_data, c_static_repeat],
                axis=1
            )  # → [N, 4 + 3 + 2 = 9]

            scaler = MinMaxScaler(feature_range=(0, 1))
            full_norm = scaler.fit_transform(full_data)

            all_scalers.append(scaler)

            # 3) Separamos nuevamente las partes normales
            accel_norm     = full_norm[:, :4]            # [N,4]
            c_time_norm    = full_norm[:, 4:7]           # [N,3]
            c_static_norm  = full_norm[:, 7:]            # [N,2]


            # -------- 3) LEER CONDICIONALES ESTÁTICAS --------
            # meta['weight'], meta['distance'] deberían ser escalares
            weight   = float(np.array(meta['weight'][:]).reshape(-1)[0])
            distance = float(np.array(meta['distance'][:]).reshape(-1)[0])
            c_static_vec = np.array([weight, distance], dtype=np.float32)  # [2]

            # -------- 4) CREAR VENTANAS --------
            for j in range(0, N - seq_len, step):

                x_win      = accel_norm[j:j+seq_len, :]      # [seq_len, 4]
                c_time_win = c_time_norm[j:j+seq_len, :]     # [seq_len, 3]

                X_list.append(x_win.astype(np.float32))
                C_time_list.append(c_time_win.astype(np.float32))
                C_static_list.append(c_static_norm[0])        # misma cond. para toda la secuencia

                if max_sequences and len(X_list) >= max_sequences:
                    break

            if max_sequences and len(X_list) >= max_sequences:
                break

    print(f"📌 Total sequences: {len(X_list)} | Each: {seq_len}×{X_list[0].shape[1]}")

    return X_list, C_time_list, C_static_list, all_scalers, feature_names





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

def batch_generator_conditional(X, C_time, C_static, time, batch_size):

    no = len(X)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    # --- X y C_time normales ---
    X_mb      = np.asarray([X[i] for i in train_idx], dtype=np.float32)
    C_time_mb = np.asarray([C_time[i] for i in train_idx], dtype=np.float32)

    # --- C_static debe quedar [B, 1, 2] ---
    C_static_mb = np.asarray([C_static[i] for i in train_idx], dtype=np.float32)

    # Si C_static_mb es (B,2), lo expandimos a (B,1,2)
    if C_static_mb.ndim == 2:
        C_static_mb = C_static_mb[:, np.newaxis, :]

    T_mb = np.asarray([time[i] for i in train_idx], dtype=np.int32)

    return X_mb, C_time_mb, C_static_mb, T_mb
