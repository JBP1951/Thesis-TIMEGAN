"""
TimeGAN model adapted for multivariate vibration signals (accelerometers, RPM, temperature, etc.) - JORFAN BALDOCEDA

Reference:
Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks," NeurIPS, 2019.

Adapted version by [Tu Nombre]
Date: [coloca la fecha actual]

This module defines:
(1) Encoder
(2) Recovery
(3) Generator
(4) Supervisor
(5) Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.init as init
#We add this line
from torch.nn.utils import spectral_norm


# -------------------------------
# Weights initialization
# -------------------------------
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


# -------------------------------
# 1. Encoder
# -------------------------------
class Encoder(nn.Module):
    """
    Embedding network: from original space → latent space
    Input:  X (seq_len, batch, z_dim)
    Output: H (seq_len, batch, hidden_dim)
    """
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
        self.ln = nn.LayerNorm(opt.hidden_dim)   # NEW
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, x, sigmoid=True):
        e_outputs, _ = self.rnn(x)        # GRU output
        e_outputs = self.ln(e_outputs)    # NEW
        H = self.fc(e_outputs)            # project
        if sigmoid:
            H = self.sigmoid(H)
        return H


# -------------------------------
# 2. Recovery
# -------------------------------
class Recovery(nn.Module):
    """
    Recovery network: latent → original feature space
    Input:  H (seq_len, batch, hidden_dim)
    Output: X_tilde (seq_len, batch, z_dim)
    """
    def __init__(self, opt):
        super(Recovery, self).__init__()
        # corregido: mantener hidden_dim en RNN
        self.rnn = nn.GRU(input_size=opt.hidden_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer,batch_first=True)
        self.ln = nn.LayerNorm(opt.hidden_dim)     # NEW
        self.fc = nn.Linear(opt.hidden_dim, opt.z_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, h, sigmoid=True):
        r_outputs, _ = self.rnn(h)
        r_outputs = self.ln(r_outputs)             # NEW
        X_tilde = self.fc(r_outputs)
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde


# -------------------------------
# 3. Generator
# -------------------------------
class Generator(nn.Module):
    """
    Generator: noise → latent space
    Input:  Z (seq_len, batch, z_dim)
    Output: E (seq_len, batch, hidden_dim)
    """
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
        self.ln = nn.LayerNorm(opt.hidden_dim)     # NEW
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, z, sigmoid=True):
        g_outputs, _ = self.rnn(z)
        g_outputs = self.ln(g_outputs)             # NEW
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E


# -------------------------------
# 4. Supervisor
# -------------------------------
class Supervisor(nn.Module):
    """
    Supervisor: predict next latent sequence from current
    Input:  H (seq_len, batch, hidden_dim)
    Output: S (seq_len, batch, hidden_dim)
    """
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(input_size=opt.hidden_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer)
        self.ln = nn.LayerNorm(opt.hidden_dim)     # NEW
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, h, sigmoid=True):
        s_outputs, _ = self.rnn(h)
        s_outputs = self.ln(s_outputs)             # NEW
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


# -------------------------------
# ⚖️ 5. Discriminator
# -------------------------------
class Discriminator(nn.Module):
    """
    Critic for WGAN-GP:
    Input:  H (seq_len, batch, hidden_dim)
    Output: critic score (seq_len, batch, 1)  -- NO sigmoid
    """
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        # GRU encoder for temporal structure
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layer
        )

        # NEW: LayerNorm after the GRU (recommended in the paper)
        self.ln = nn.LayerNorm(opt.hidden_dim)

        # IMPORTANT: for WGAN-GP we must REMOVE spectral norm and sigmoid.
        self.fc = nn.Linear(opt.hidden_dim, 1)

        self.apply(_weights_init)

    def forward(self, h, sigmoid: bool = False):
        """
        h: [seq_len, batch, hidden_dim]
        """
        with torch.backends.cudnn.flags(enabled=False):
            d_outputs, _ = self.rnn(h)
       
        d_outputs = self.ln(d_outputs)       # layer normalization (NEW)

        logits = self.fc(d_outputs)          # raw critic score

        # sigmoid is ignored because WGAN uses raw scores
        return logits
        
    ''' self.fc = nn.Linear(opt.hidden_dim, 1)  # <- corregido para probabilidad
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, h, sigmoid=True):
        d_outputs, _ = self.rnn(h)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat'''