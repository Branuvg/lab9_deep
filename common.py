"""Código compartido del Laboratorio 9 (extraído de lab9.ipynb, Tasks 1 y 2).

Uso desde cualquier notebook de la carpeta:
    from common import *
    X_train, Y_train, X_val, Y_val = load_fashion(device)        # en [0,1]
    X_train_d, X_val_d = to_diffusion_range(X_train), to_diffusion_range(X_val)   # en [-1,1]
"""
import math, random
import numpy as np
import torch
import torch.nn as nn

SEED = 42
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- datos (Task 1.1)
def load_fashion(device):
    """Fashion-MNIST en [0,1]; validación = últimos 5 000 ejemplos de entrenamiento."""
    from torchvision import datasets
    raw = datasets.FashionMNIST("data", train=True, download=True)
    X = raw.data.float().div(255.).unsqueeze(1)          # (60000,1,28,28)
    Y = raw.targets.clone()
    return (X[:-5000].to(device), Y[:-5000].to(device),
            X[-5000:].to(device), Y[-5000:].to(device))


def to_diffusion_range(x):
    """[0,1] -> [-1,1] (Task 2.1)."""
    return x * 2 - 1


# ---------------------------------------------------------------- calendario lineal (Task 2.1)
T = 1000
beta_start = 1e-4
beta_end = 0.02


def linear_schedule(T=T, beta_start=beta_start, beta_end=beta_end):
    """Devuelve (betas, alphas, alpha_bar) en float64; el índice t-1 corresponde al instante t = 1..T."""
    t = torch.arange(1, T + 1, dtype=torch.float64)
    betas = beta_start + (t - 1) / (T - 1) * (beta_end - beta_start)   # beta_t lineal
    alphas = 1.0 - betas                                               # alpha_t = 1 - beta_t
    alpha_bar = torch.cumprod(alphas, dim=0)                           # alpha_bar_t = prod alpha_s
    return betas, alphas, alpha_bar


def q_sample(x0, t, eps, alpha_bar):
    """Forma cerrada del forward: x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps.
    x0, eps: (B,1,28,28); t: (B,) enteros en {1..T}; alpha_bar: tensor en el mismo device."""
    ab = alpha_bar[t - 1].view(-1, 1, 1, 1).to(x0.dtype)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps


# ---------------------------------------------------------------- VAE (Task 1.2)
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7), nn.ReLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.dec(self.dec_fc(z).view(-1, 64, 7, 7))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparametrización
        return self.decode(z), mu, logvar


def load_vae(beta, device):
    """Carga un VAE entrenado en la Task 1.2 (beta en {0.1, 1.0, 10.0}). Sus salidas están en [0,1]."""
    m = VAE().to(device)
    m.load_state_dict(torch.load(f"checkpoints/vae_beta{float(beta)}.pt", map_location=device))
    return m.eval()


# ---------------------------------------------------------------- métrica de nitidez (Task 1.4d)
def sharpness(x):
    """Magnitud media de la diferencia entre píxeles horizontalmente adyacentes.
    Usar siempre imágenes en [0,1] para comparar con la Task 1.4(d) (difusión: (x+1)/2)."""
    return (x[..., :, 1:] - x[..., :, :-1]).abs().mean().item()
