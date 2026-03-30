from tqdm import tqdm
import os
import torch
import torch.optim as optim
import torch.nn as nn
from Diffusion_and_Vae_components import device,VAE,DiffConfig
from Dataloader import train_dataloader

VAE_model=VAE().to(device)
epochs=DiffConfig.VAE_epochs
optimizer=optim.Adam(VAE_model.parameters(),lr=DiffConfig.VAE_learning_rate)
loss_fn=nn.MSELoss()

if os.path.exists('/kaggle/working/vae_with_Residual_and_annnealing_checkpoint.pth'):
    checkpoint=torch.load('/kaggle/working/vae_with_Residual_and_annnealing_checkpoint.pth')
    VAE_model.load_state_dict(checkpoint['model_state_dict'])
    print('model found')

save_dir = '/kaggle/working/'
start_epoch = 0
target_kl_weight=DiffConfig.kl_weight
anneal_steps=10000
current_global_step=start_epoch*len(train_dataloader)
for i in tqdm(range(start_epoch, epochs)):
    for batch, (train_features_batch, _) in enumerate(train_dataloader):
        input_image=train_features_batch.to(torch.float32).to(device)
        # print(f'input image {input_image.shape}')
        log_var,mean,predicted_image=VAE_model(input_image)
        # print(f'log_var,mean,predicted_image {log_var.shape},{mean.shape},{predicted_image.shape}')
        loss_1=loss_fn(predicted_image,input_image)
        loss_2=-0.5*torch.mean(1+log_var-(mean**2)-torch.exp(log_var))
        current_kl_weight = min(target_kl_weight, 
                               target_kl_weight * (current_global_step / anneal_steps))
        loss=loss_1+(current_kl_weight*loss_2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch%300==0:
            print(f'Epoch {i} Batch {batch} loss {loss.item()}')
    
    if i % 1 == 0:
        torch.save({
            'epoch': i,
            'model_state_dict': VAE_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, f'{save_dir}vae_with_Residual_and_annnealing_checkpoint.pth')
        print(f"Saved checkpoint at epoch {i}")
