class Convolution_Double(nn.Module):
  def __init__(self,input_channels:int,hidden_channels:int):
    super().__init__()
    self.conv_block=nn.Sequential(
        nn.Conv2d(in_channels=input_channels,
                  out_channels=hidden_channels,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.GroupNorm(8,hidden_channels),
        nn.SiLU(),

        
        nn.Conv2d(in_channels=hidden_channels,
                  out_channels=hidden_channels,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.GroupNorm(8,hidden_channels),
        nn.SiLU(),
        )
  def forward(self,x):
    x=self.conv_block(x)
    return x

class Decoder(nn.Module):
  def __init__(self,in_channels:int,out_channels:int):
    super().__init__()
    self.upsample_layer=nn.ConvTranspose2d(in_channels=in_channels,out_channels=in_channels,kernel_size=2,stride=2)
    self.conv_block=Convolution_Double(2*in_channels,out_channels)
  def forward(self,x,skip_connection_input):
    x=self.upsample_layer(x)
    shape=x.shape
    skip_connection_input=skip_connection_input[:,:,0:shape[2],0:shape[3]]
    x=torch.cat((x,skip_connection_input),dim=1)

    x=self.conv_block(x)

    return x
 class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(x + self.block(x)) 
