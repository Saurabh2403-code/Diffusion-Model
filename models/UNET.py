class PositionalEncoding(nn.Module):
    """
    Docstring for PositionalEncoding
    This is a subclass of nn.Module.Takes input as bandwidth(L_pos or L_dir) which takes a set of co ordinates and maps it into
    higher dimension.
    Eg: x-->(x,sin(x),cos(x),sin(2x),cos(2x),sin((2^2)x),cos((2^2)x)....sin((2^l-1)x),cos((2^l-1)x))
    So 3d dimensional coordinate=39 dimensional coordinate(L=6)
    """
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth

        self.register_buffer('frequencies', 2.0 ** torch.linspace(0.0, bandwidth - 1, bandwidth) * np.pi) #stores a tensor named 'frequencies' on gpu or mps

    def forward(self, x):
        out = [x]
        for frequency in self.frequencies:
            out.append(torch.sin(x * frequency))
            out.append(torch.cos(x * frequency))
        return torch.cat(out, dim=-1)

class time_encoder(nn.Module):
    def __init__(self,bandwidth:int,out_dimension):
        super().__init__()
        self.encoder=PositionalEncoding(bandwidth)
        self.linear_layer=nn.Linear((2*bandwidth+1),out_dimension)
        self.Silu=nn.SiLU()
        self.linear_l2=nn.Linear(out_dimension,out_dimension)

    def forward(self,x:torch.tensor):
        x=x.unsqueeze(-1)
        x=self.encoder(x)
        x=self.Silu(self.linear_layer(x))
        return self.linear_l2(x).squeeze(1)

class UNET(nn.Module):
  def __init__(self):
    super().__init__()
    self.time_encoder=time_encoder(20,256)

    self.time_proj_0=nn.Linear(256,64)
    self.time_proj_1=nn.Linear(256,128)
    # self.time_proj_2=nn.Linear(256,256)

    self.encoder_1=Convolution_Double(4,64)
    self.encoder_2=Convolution_Double(64,128)
    # self.encoder_3=Convolution_Double(128,256)

    self.decoder_3=Decoder(128,64)
    self.decoder_2=Decoder(64,32)
    self.output_conv=nn.Conv2d(32,4,kernel_size=1) 

    self.m=nn.MaxPool2d(2)


  def forward(self,x,t):
    # print(f't {t.shape}')
    t_encoded=self.time_encoder(t)#this return me a tensor of [B,256]
    # print(f't_encoded {t_encoded.shape}')
    t_projected_0=self.time_proj_0(t_encoded)[...,None,None]
    # print(f't_projected_0 {t_projected_0.shape}')
    

    x_t_0=x
    # print(f'x_t_0 shape {x_t_0.shape}')
    skip_0=self.encoder_1(x_t_0)
    # print(f'skip_0 {skip_0.shape}')
    skip_0=skip_0+t_projected_0
    out_1=self.m(skip_0)
    # print(f'out_1 {out_1.shape}')

    t_projected_1=self.time_proj_1(t_encoded)[...,None,None]
    # print(f't_projected{t_projected_1.shape}')
    x_t_1=out_1
    # print(f'x_t_1 {x_t_1.shape}')
    skip_1=self.encoder_2(x_t_1)
    # print(f'skip_1 {skip_1.shape}')
    skip_1=skip_1+t_projected_1
    out_2=self.m(skip_1)
    # print(f'out_2 {out_2.shape}')

    # t_projected_2=self.time_proj_2(t_encoded)[...,None,None]
    # x_t_3=out_2
    # skip_2=self.encoder_3(x_t_3)
    # skip_2=skip_2+t_projected_2
    # out_3=skip_2

    ''' Now upsampling'''
    out_t_2=self.decoder_3(out_2,skip_1)#[B,256,:,:]--->[B,128,:,:]
    # print(f'out_t_2 {out_t_2.shape}')
    out_t_1=self.decoder_2(out_t_2,skip_0)#[B,128,:,:--->[B,64,:,:,]
    # print(f'out_t_1 {out_t_1.shape}')
    out=self.output_conv(out_t_1)
    # print(f'out {out.shape}')

    return out
