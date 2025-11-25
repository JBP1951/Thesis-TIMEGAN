"""
Train_TGAN.py — Adapted for Thesis JORDAN BALDOCEDA

Based on:
"Time-series Generative Adversarial Networks"
Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, NeurIPS 2019.

Adapted by [JORDAN BALDOCEDA] for:
- Rotor dynamics & SHM signals (accelerometers, temperature, encoder)
- Integration with custom surrogate model & experimental data

-----------------------------
Main Workflow:
(1) Load dataset (your signals)
(2) Initialize and train TimeGAN
(3) Optionally visualize or export synthetic signals
"""

import warnings
warnings.filterwarnings("ignore")

import torch
from opt_parser import opt_parser  # your adapted options parser
from lib.data import load_data     # your data loading logic (to be adapted if needed)

from data_loader_conditional import load_all_experiments
from lib.timegan import TimeGAN    # your adapted TimeGAN model
from options_TGAN import Options


def train_TGAN():
    """Main training function for thesis-adapted TimeGAN."""

    # Parse arguments (safe for Jupyter or script)
    try:
        opt = opt_parser.parser.parse_args(args=[])  # Works in Jupyter
    except Exception:
        opt = opt_parser.parser.parse_args()         # Works in terminal

    print(" Options loaded successfully.")

    # Load your experimental or surrogate dataset
        # ===============================================
    # CONDITIONAL DATA LOADING (REEMPLAZA AL ORIGINAL)
    # ===============================================


    print("📌 Loading conditional structured dataset ...")

    # carga tus experimentos del archivo .mat
    X_list, C_time_list, C_static_list = load_all_experiments(mat_path=mat_file,
        seq_len=opt.seq_len,
        step=1,
        max_sequences=None )

    # Construimos estructura TimeGAN-condicional
    ori_data = {
        "X": X_list,            # lista de matrices (L,4)
        "C_time": C_time_list,  # lista de matrices (L,3)
        "C_static": C_static_list  # lista de vectores (2,)
    }

    print("✔ Datos cargados correctamente")
    print(f"   N experimentos: {len(X_list)}")
    print(f"   Ejemplo shapes:")
    print(f"     X:        {X_list[0].shape}")
    print(f"     C_time:   {C_time_list[0].shape}")
    print(f"     C_static: {C_static_list[0].shape}")


    # Initialize TimeGAN model
    model = TimeGAN(opt, ori_data)
    print(" TimeGAN model initialized successfully.")

    # Train model
    print(" Starting TimeGAN training...")
    model.train()
    print("Training completed.")

    '''#  (Optional) Generate synthetic data after training
    generated_data = model.generation(num_samples=opt.batch_size)
    print(f" Generated synthetic samples: {len(generated_data)}")

    # Optional save: you can uncomment if you want to store the generated dataset
    # import numpy as np
    # np.save("generated_TGAN_data.npy", generated_data)
    '''
    return model


if __name__ == "__main__":
    train_TGAN()
