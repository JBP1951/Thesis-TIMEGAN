import numpy as np
import h5py

def load_all_experiments(
    mat_path,
    seq_len=256,
    step=1,
    max_sequences=None
):
    """
    Carga el archivo all_signals_processed.mat y separa:
      - X        : [Accel1..Accel4]
      - C_time   : [Motor_current, Speed, Temperature]
      - C_static : [weight, distance]

    Devuelve 3 listas:
      X_list        -> cada elemento: [seq_len, 4]
      C_time_list   -> cada elemento: [seq_len, 3]
      C_static_list -> cada elemento: [2]
    """

    X_list = []
    C_time_list = []
    C_static_list = []

    with h5py.File(mat_path, "r") as f:
        data_all = f["data_all"]
        n_files = data_all.shape[1]

        print(f"📂 Experimentos en data_all: {n_files}")

        for i in range(n_files):

            ref = data_all[0][i]
            entry = f[ref]

            sig_group = entry["signals_processed"]
            meta_group = entry["meta"]

            # --- 1) Señales: accel y condiciones de tiempo ---
            accel_names = ["Accel1", "Accel2", "Accel3", "Accel4"]
            cond_time_names = ["Motor_current", "Speed", "Temperature"]

            # Accels → 4 columnas
            accels = []
            for name in accel_names:
                vec = np.array(sig_group[name][:]).reshape(-1)
                accels.append(vec)
            X_full = np.vstack(accels).T.astype(np.float32)   # [N, 4]

            # Condiciones de tiempo → 3 columnas
            ctime = []
            for name in cond_time_names:
                vec = np.array(sig_group[name][:]).reshape(-1)
                ctime.append(vec)
            C_time_full = np.vstack(ctime).T.astype(np.float32)  # [N, 3]

            # --- 2) Condicionales estáticas (weight, distance) ---
            weight = float(np.array(meta_group["weight"])[0][0])
            distance = float(np.array(meta_group["distance"])[0][0])
            C_static_vec = np.array([weight, distance], dtype=np.float32)

            # --- 3) Crear ventanas deslizantes ---
            N = X_full.shape[0]
            for start in range(0, N - seq_len, step):
                end = start + seq_len

                X_seq = X_full[start:end, :]           # [seq_len, 4]
                C_time_seq = C_time_full[start:end, :] # [seq_len, 3]

                X_list.append(X_seq)
                C_time_list.append(C_time_seq)
                C_static_list.append(C_static_vec)

                if max_sequences is not None and len(X_list) >= max_sequences:
                    break

            if max_sequences is not None and len(X_list) >= max_sequences:
                break

    print(f"✅ Total ventanas creadas: {len(X_list)} | seq_len={seq_len}")
    print(f"   Ejemplo X      : {X_list[0].shape}")
    print(f"   Ejemplo C_time : {C_time_list[0].shape}")
    print(f"   Ejemplo C_stat : {C_static_list[0].shape}")

    return X_list, C_time_list, C_static_list
