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
    ori_data = load_data(opt)
    print(f" Original data loaded: shape = {len(ori_data)} sequences")

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
