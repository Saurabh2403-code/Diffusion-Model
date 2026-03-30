import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import tqdm
from Diffusion_and_Vae_components import UNET,device,DiffConfig,VAE,noise_scheduler
from DiffusionConfig import DiffConfig
from Vae_training import VAE_model
from torch.utils.data import DataLoader

net=UNET().to(torch.float32).to(device)

wepochs=DiffConfig.DDPM_epochs
optimizer=optim.Adam(net.parameters(),lr=DiffConfig.DDPM_learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
loss_fn=nn.MSELoss()
with torch.inference_mode():
    sample_batch, _ = next(iter(train_dataloader))
    sample_encoded = VAE_model.encoder(sample_batch.to(device))
    _, sample_mean, _ = VAE_model.latent_sampler(sample_encoded)
    print(f"Latent mean: {sample_mean.mean():.4f}")
    print(f"Latent std:  {sample_mean.std():.4f}")
    # Should be close to mean≈0, std≈1

all_means = []
all_stds = []

VAE_model.eval()
with torch.inference_mode():
    for batch, (imgs, _) in tqdm(enumerate(train_dataloader)):
        if batch > 100:  # sample 100 batches
            break
        encoded = VAE_model.encoder(imgs.to(device))
        _, mean, _ = VAE_model.latent_sampler(encoded)
        all_means.append(mean.mean().item())
        all_stds.append(mean.std().item())

mean_global = torch.tensor(sum(all_means)/len(all_means)).to(device)
std_global = torch.tensor(sum(all_stds)/len(all_stds)).to(device)

if os.path.exists('/kaggle/working/CelebA_with_res_net_in_vae_with_wider_ddpm.pth'):
    print('model found')
    checkpoint=torch.load('/kaggle/working/CelebA_with_res_net_in_vae_with_wider_ddpm.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
        
for param in VAE_model.parameters():
    param.requires_grad = False
VAE_model.eval()

for i in tqdm(range(epochs)):
    net.train()  
    loss_per_epoch = []
    
    for batch, (train_features_batch, _) in enumerate(train_dataloader):
        with torch.inference_mode():
            encoded = VAE_model.encoder(train_features_batch.to(torch.float32).to(device))
            _, mean, _ = VAE_model.latent_sampler(encoded)
                
        x_0=(mean-mean_global)/std_global
        out=noise_scheduler(x_0)
        
        corrupted_image=out[0].to(torch.float32).to(device)
      # print(f'corrupted image {corrupted_image.shape}')
        t_normalized=(out[2]/1000).to(torch.float32)
        t_normalized=t_normalized.squeeze(-1).to(device)
      # print(f't_normalized {t_normalized.shape}')
        epsilon=out[1].to(torch.float32).to(device)
      # print(f' epsilon {epsilon.shape}')
        predicted_noise=net(corrupted_image,t_normalized)
      # print(f'predicted_noise.shape')
      # print(predicted_noise.mean(),predicted_noise.std())
      # print(epsilon.mean(),epsilon.std())
        loss=loss_fn(predicted_noise,epsilon)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
      # print(device)
      # print(next(net.parameters()).device)
        if batch%1000==0:
          print(f"epoch{i} batch {batch} loss {loss.item()}")
        if batch%2000==0:
          torch.save(
              {
              'epoch': i,
              'batch':batch,
              'model_state_dict': net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'scheduler_state_dict': scheduler.state_dict(),
              'loss': loss.item()
               }, '/kaggle/working/CelebA_with_res_net_in_vae_with_wider_ddpm.pth')
    scheduler.step()
    print(f'loss per epoch{loss.item()}')
    loss_per_epoch.append(loss.item())
    torch.save({
        'epoch': i,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
    }, '/kaggle/working/CelebA_with_res_net_in_vae_with_wider_ddpm.pth')
