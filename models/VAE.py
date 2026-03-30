class VAE_Encoder(nn.Module):
  def __init__(self,input_channels,hidden_channels,output_channels):
    super().__init__()
    self.encoder_layer=nn.Sequential(
        nn.Conv2d(in_channels=input_channels,
                  out_channels=hidden_channels,
                  kernel_size=3,stride=1,padding=1),
        nn.GroupNorm(8,hidden_channels),
        nn.SiLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(in_channels=hidden_channels,out_channels=2*(hidden_channels),kernel_size=3,stride=1,padding=1),
        nn.GroupNorm(8,2*hidden_channels),
        nn.SiLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(in_channels=2*hidden_channels,
                  out_channels=output_channels,
                  kernel_size=3,
                  stride=1,padding=1),
        nn.GroupNorm(8,output_channels),
        nn.SiLU(),
        nn.MaxPool2d(2)

    )
  def forward(self,x):
    return self.encoder_layer(x)


class VAE_Latent(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,x):
    channel=int(x.shape[1]/2)
    mean=x[:,:channel,:,:]
    log_var=x[:,channel:,:,:]
    std=torch.exp(0.5*log_var)
    noise=torch.randn_like(std)
    sampled_z=mean+torch.mul(std,noise)
    return log_var,mean,sampled_z


class VAE_Decoder(nn.Module):
  def __init__(self, in_channels, hidden_channels):
    super().__init__()
    self.decoder_layer = nn.Sequential(

        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(8, hidden_channels),
        nn.SiLU(),
        nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, hidden_channels),
        nn.SiLU(),

  
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(8, hidden_channels),
        nn.SiLU(),
        nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, hidden_channels),
        nn.SiLU(),

        # Level 3: 64x64 → 128x128
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(8, hidden_channels),
        nn.SiLU(),
        nn.ConvTranspose2d(hidden_channels, 3, kernel_size=4, stride=2, padding=1),
        nn.Tanh()
    )

  def forward(self, x):
    return self.decoder_layer(x)
  

class VAE(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder=VAE_Encoder(3,64,DiffConfig.VAE_latent_out_channels)
    self.latent_sampler=VAE_Latent()
    self.decoder=VAE_Decoder(DiffConfig.VAE_latent_out_channels/2,256)
  def forward(self,x):
    # print(f'input {x.shape}')
    x=self.encoder(x)
    # print(f'encoded {x.shape}')
    log_var,mean,latent_image=self.latent_sampler(x)
    # print(f'latent_vector {latent_vector.shape} latent image {latent_image.shape}')
    x=self.decoder(latent_image)
    # print(f'decoded {x.shape}')
    return log_var,mean,x
