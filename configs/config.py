from dataclasses import dataclass
import torch
@dataclass
class DiffConfig:
    batch_size=128
    VAE_epochs=100
    DDPM_epochs=10
    VAE_learning_rate=1e-4
    DDPM_learning_rate=1e-4
    VAE_latent_out_channels=16
    position_encoding_bandwidth=16
    x_intial,x_final,y_intial,y_final=0,1,1e-4,0.02
    time_steps=1000
    kl_weight=1e-4
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
