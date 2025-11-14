"""
timegan_TGAN.py
Adapted (minimal changes) TimeGAN runner based on the PyTorch reimplementation.

Keep structure and method names identical to original; minimal adaptions:
 - imports adjusted to local utils/data/module names used in this thesis
 - robust device detection
 - safe renormalization in `generation()`

 Note: Use original data as training set to generater synthetic data (time-series)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Use your data_preprocess batch generator (must provide batch_generator)
from lib.data_preprocess import batch_generator

# Use the utils file adapted for your repo (utils_TGAN.py)
from utils_TGAN import extract_time, random_generator, NormMinMax

# Import the adapted model (model_TGAN.py)
from lib.model_TGAN import Encoder, Recovery, Generator, Discriminator, Supervisor


class BaseModel():
  """ Base Model for TimeGAN (kept same structure as original) """
  def __init__(self, opt, ori_data):
    # Seed for deterministic behavior
    self.seed(getattr(opt, "manualseed", -1))

    # Initialize variables
    self.opt = opt

    # Normalize original data (expects ori_data as list/array of sequences)
    self.ori_data, self.min_val, self.max_val = NormMinMax(ori_data)

    #THIS LINE WAS ADDED
            # ✅ Automatically adapt latent and input dimensions to data
    first_seq = np.asarray(self.ori_data[0])
    self.opt.z_dim = first_seq.shape[1]
    print(f"[INFO] Adjusted opt.z_dim to match data feature size: {self.opt.z_dim}")


    # Extract times and maximum sequence length
    self.ori_time, self.max_seq_len = extract_time(self.ori_data)

    # Data count (assumes ori_data convertible to numpy array of shape [N, seq_len, feat])
    self.data_num, _, _ = np.asarray(ori_data).shape

    # Train/test directories for checkpoints/logs
    self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
    self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

    # Device (use GPU if requested and available)
    if getattr(self.opt, "device", "gpu") != 'cpu' and torch.cuda.is_available():
      self.device = torch.device("cuda:0")
    else:
      self.device = torch.device("cpu")

    # ADDED TO SOLVE DEBUGGING PART OF LOSS
    # losses (same as original)
    self.l_mse = nn.MSELoss()
    self.l_r = nn.L1Loss()
    self.l_bce = nn.BCELoss()


  

  def seed(self, seed_value):
    """Seed RNGs for reproducibility"""
    if seed_value == -1:
      return
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True

  def save_weights(self, epoch):
    """Save nets' weights for current epoch"""
    weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
    if not os.path.exists(weight_dir):
      os.makedirs(weight_dir)

    torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()},
               '%s/netE.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netr.state_dict()},
               '%s/netR.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
               '%s/netG.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
               '%s/netD.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.nets.state_dict()},
               '%s/netS.pth' % (weight_dir))


  def train_one_iter_er(self):
    """ Train encoder & recovery for one mini-batch """
    self.nete.train()
    self.netr.train()

    # set mini-batch
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

    # train encoder & decoder
    loss = self.optimize_params_er()
    return loss


  def train_one_iter_er_(self):
    """ Train encoder & recovery (alternate objective) for one mini-batch """
    self.nete.train()
    self.netr.train()

    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

    self.optimize_params_er_()


  def train_one_iter_s(self):
    """ Train supervisor """
    self.nets.train()
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.optimize_params_s()


  def train_one_iter_g(self):
    """ Train generator-related parts """
    self.netg.train()
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)
    self.Z = np.asarray(self.Z)  # make sure numpy array
    self.optimize_params_g()


  def train_one_iter_d(self):
    """ Train discriminator """
    self.netd.train()
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)
    self.Z = np.asarray(self.Z)
    self.optimize_params_d()


  def train(self):
    """High-level training loop (same flow as original)"""
    for it in range(self.opt.iteration):
      self.train_one_iter_er()
      print('Encoder training step: {}/{}'.format(it, self.opt.iteration))

    for it in range(self.opt.iteration):
      self.train_one_iter_s()
      print('Supervisor training step: {}/{}'.format(it, self.opt.iteration))

    # we modify this beucase WGAN  need to train 
    for it in range(self.opt.iteration):

      # 1) Train critic (discriminator) n_critic times
      for _ in range(self.opt.n_critic):
          self.train_one_iter_d()

      # 2) Train generator + supervisor once
      self.train_one_iter_g()

      print(f'WGAN adversarial step: {it}/{self.opt.iteration}')



    '''for it in range(self.opt.iteration):
      for kk in range(2):
        self.train_one_iter_g()
        self.train_one_iter_er_()
      self.train_one_iter_d()
      print('Adversarial training step: {}/{}'.format(it, self.opt.iteration))
    '''
    self.save_weights(self.opt.iteration)
    self.generated_data = self.generation(self.opt.batch_size)
    print('Finish Synthetic Data Generation')


  def generation(self, num_samples, mean = 0.0, std = 1.0):
    """Generate synthetic sequences (num_samples) and renormalize to original scale"""
    if num_samples == 0:
      return None

    # get a mini-batch (to obtain T info)
    self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)

    # create random noise for generation
    self.Z = random_generator(num_samples, self.opt.z_dim, self.T, self.max_seq_len, mean, std)
    self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)

    # forward pass through generator + supervisor + recovery
    self.E_hat = self.netg(self.Z)       # [num_samples, seq_len, hidden_dim] or (seq_len, batch, hidden)
    self.H_hat = self.nets(self.E_hat)
    
    generated_data_curr = self.netr(self.H_hat).cpu().detach().numpy()  # shape: [num_samples, seq_len, z_dim]

    # slice each sample to its original length and collect
    generated_data = []
    for i in range(num_samples):
      temp = generated_data_curr[i, :self.ori_time[i], :]
      generated_data.append(temp)

    # Renormalization: apply per-feature min/max to each sequence
    # min_val and max_val are arrays (features,)
    # ensure broadcasting works: (seq_len, features) * (features,) -> (seq_len, features)
    generated_data = [ (seq * self.max_val) + self.min_val for seq in generated_data ]

    return generated_data



class TimeGAN(BaseModel):
    """TimeGAN class (keeps original flow and method names)"""

    @property
    def name(self):
      return 'TimeGAN'

    def __init__(self, opt, ori_data):
      super(TimeGAN, self).__init__(opt, ori_data)

      # -- Misc attributes
      self.epoch = 0
      self.times = []
      self.total_steps = 0

      # Create and initialize networks.
      self.nete = Encoder(self.opt).to(self.device)
      self.netr = Recovery(self.opt).to(self.device)
      self.netg = Generator(self.opt).to(self.device)
      self.netd = Discriminator(self.opt).to(self.device)
      self.nets = Supervisor(self.opt).to(self.device)

      # Optionally load pre-trained models (kept same)
      if getattr(self.opt, "resume", '') != '':
        print("\nLoading pre-trained networks.")
        self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
        self.nete.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netE.pth'))['state_dict'])
        self.netr.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netR.pth'))['state_dict'])
        self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
        self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
        self.nets.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netS.pth'))['state_dict'])
        print("\tDone.\n")

      # losses (same as original)
      self.l_mse = nn.MSELoss()
      self.l_r = nn.L1Loss()
      self.l_bce = nn.BCELoss()

      # Setup optimizer (requires opt.lr and opt.beta1 — keep them in options)
      if getattr(self.opt, "isTrain", True):
        self.nete.train()
        self.netr.train()
        self.netg.train()
        self.netd.train()
        self.nets.train()

        # allow fallback defaults if not present in opt
        lr = getattr(self.opt, "lr", 0.001)
        beta1 = getattr(self.opt, "beta1", 0.9)

        self.optimizer_e = optim.Adam(self.nete.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_r = optim.Adam(self.netr.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_s = optim.Adam(self.nets.parameters(), lr=lr, betas=(beta1, 0.999))


    # -----------------------
    # Forward helpers (same names as original)
    # -----------------------
    def forward_e(self):
      self.H = self.nete(self.X)

    def forward_er(self):
      self.H = self.nete(self.X)
      self.X_tilde = self.netr(self.H)

    def forward_g(self):
      self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
      self.E_hat = self.netg(self.Z)

    def forward_dg(self):
      self.Y_fake = self.netd(self.H_hat)
      self.Y_fake_e = self.netd(self.E_hat)

    def forward_rg(self):
      self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
      self.H_supervise = self.nets(self.H)

    def forward_sg(self):
      self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
      self.Y_real = self.netd(self.H)
      self.Y_fake = self.netd(self.H_hat)
      self.Y_fake_e = self.netd(self.E_hat)


    # -----------------------
    # Backward helpers (same as original)
    # -----------------------
    def backward_er(self):
      self.err_er = self.l_mse(self.X_tilde, self.X)
      self.err_er.backward(retain_graph=True)
      print("Loss: ", self.err_er)

    def backward_er_(self):
      self.err_er_ = self.l_mse(self.X_tilde, self.X)
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
      self.err_er.backward(retain_graph=True)

    def backward_g(self):
      #Here we also reeplace the function

       
      adv_fake = - self.Y_fake.mean()
      adv_fake_e = - self.Y_fake_e.mean()
      self.err_g_adv = adv_fake + self.opt.w_gamma * adv_fake_e

      # --- Regularizadores estadísticos que ya usabas ---
      # CUIDADO: X_hat y X pueden tener shape [seq, batch, feat] o [batch, seq, feat].
      # Tus líneas actuales asumen índice [0] y [1] como (stat, time?) → no las toco para mantener compatibilidad.
      self.err_g_V1 = torch.mean(torch.abs(
          torch.sqrt(torch.std(self.X_hat, [0])[1] + 1e-6) -
          torch.sqrt(torch.std(self.X,     [0])[1] + 1e-6)
      ))
      self.err_g_V2 = torch.mean(torch.abs(
          (torch.mean(self.X_hat, [0])[0]) -
          (torch.mean(self.X,     [0])[0])
      ))

      # --- Pérdida de supervisión (igual que antes) ---
      self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])

      # --- Total del generador (mantenemos tus pesos) ---
      self.err_g = self.err_g_adv \
                  + self.err_g_V1 * self.opt.w_g \
                  + self.err_g_V2 * self.opt.w_g \
                  + torch.sqrt(self.err_s)

      self.err_g.backward(retain_graph=True)
      print("Loss G (total): ", self.err_g)

      '''
      self.err_g_U = self.l_bce(self.Y_fake, torch.ones_like(self.Y_fake))
      self.err_g_U_e = self.l_bce(self.Y_fake_e, torch.ones_like(self.Y_fake_e))
      self.err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(self.X_hat,[0])[1] + 1e-6) - torch.sqrt(torch.std(self.X,[0])[1] + 1e-6)))
      self.err_g_V2 = torch.mean(torch.abs((torch.mean(self.X_hat,[0])[0]) - (torch.mean(self.X,[0])[0])))
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      self.err_g = self.err_g_U + \
                   self.err_g_U_e * self.opt.w_gamma + \
                   self.err_g_V1 * self.opt.w_g + \
                   self.err_g_V2 * self.opt.w_g + \
                   torch.sqrt(self.err_s)
      self.err_g.backward(retain_graph=True)
      print("Loss G: ", self.err_g)'''

    def backward_s(self):
      self.err_s = self.l_mse(self.H[:,1:,:], self.H_supervise[:,:-1,:])
      self.err_s.backward(retain_graph=True)
      print("Loss S: ", self.err_s)

    # GRADIENT PENALTY NEW
    def gradient_penalty(self, real, fake):
    #""" Compute gradient penalty for WGAN-GP in latent space H """
      alpha = torch.rand(real.size(0), 1, 1).to(self.device)
      alpha = alpha.expand_as(real)

      interpolates = alpha * real + ((1 - alpha) * fake)
      interpolates.requires_grad_(True)

      d_interpolates = self.netd(interpolates)

      gradients = torch.autograd.grad(
          outputs=d_interpolates,
          inputs=interpolates,
          grad_outputs=torch.ones_like(d_interpolates),
          create_graph=True,
          retain_graph=True,
          only_inputs=True
      )[0]

      gradients = gradients.reshape(real.size(0), -1)
      gradient_norm = gradients.norm(2, dim=1)

      gp = ((gradient_norm - 1) ** 2).mean()
      return gp



    def backward_d(self):
      #here we reeplace entirly the funciton
      """
    Hinge loss para el discriminador:

      L_D = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))] + w_gamma * E[max(0, 1 + D(fake_e))]

    Donde:
      - Y_real = D(H)               (features reales)
      - Y_fake = D(H_hat)           (features generadas por G->S)
      - Y_fake_e = D(E_hat)         (features generadas solo por G)
    """
   
      # WGAN critic loss
      loss_real = -self.Y_real.mean()
      loss_fake = self.Y_fake.mean()
      loss_fake_e = self.Y_fake_e.mean()

      # Gradient Penalty in latent space H_hat
      gp = self.gradient_penalty(self.H.detach(), self.H_hat.detach())

      self.err_d = loss_real + loss_fake + self.opt.w_gamma * loss_fake_e + self.opt.gp_lambda * gp

      self.err_d.backward(retain_graph=True)



      '''
      self.err_d_real = self.l_bce(self.Y_real, torch.ones_like(self.Y_real))
      self.err_d_fake = self.l_bce(self.Y_fake, torch.zeros_like(self.Y_fake))
      self.err_d_fake_e = self.l_bce(self.Y_fake_e, torch.zeros_like(self.Y_fake_e))
      self.err_d = self.err_d_real + \
                   self.err_d_fake + \
                   self.err_d_fake_e * self.opt.w_gamma
      if self.err_d > 0.15:
        self.err_d.backward(retain_graph=True)'''


    # -----------------------
    # Optimize wrappers (same as original)
    # -----------------------
    def optimize_params_er(self):

      self.forward_er()
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()

      #this line was added to debugging
      self.loss_er = self.l_mse(self.X_tilde, self.X)

      self.loss_er.backward(retain_graph=True)
      self.optimizer_e.step()
      self.optimizer_r.step()

      print(f"[DEBUG] Loss this iteration: {self.loss_er.item():.6f}")
      return float(self.loss_er.item())

    def optimize_params_er_(self):
      self.forward_er()
      self.forward_s()
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er_()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_s(self):
      self.forward_e()
      self.forward_s()
      self.optimizer_s.zero_grad()
      self.backward_s()
      self.optimizer_s.step()

    def optimize_params_g(self):
      self.forward_e()
      self.forward_s()
      self.forward_g()
      self.forward_sg()
      self.forward_rg()
      self.forward_dg()
      self.optimizer_g.zero_grad()
      self.optimizer_s.zero_grad()
      self.backward_g()
      self.optimizer_g.step()
      self.optimizer_s.step()

    def optimize_params_d(self):
      self.forward_e()
      self.forward_g()
      self.forward_sg()
      self.forward_d()
      self.forward_dg()
      self.optimizer_d.zero_grad()
      self.backward_d()
      self.optimizer_d.step()
