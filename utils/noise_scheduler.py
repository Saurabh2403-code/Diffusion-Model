def linear(x_intial=DiffConfig.x_intial,y_intial=DiffConfig.y_intial,x_final=DiffConfig.x_final,y_final=DiffConfig.y_final):
  beta=torch.linspace(y_intial,y_final,DiffConfig.time_steps)
  alpha=1-beta
  return alpha,beta

alpha,beta=linear()

alpha=alpha.to(torch.float32).to(device)
beta=beta.to(torch.float32).to(device)

alpha_bars=torch.cumprod(alpha,dim=0)
alpha_bars=alpha_bars.to(torch.float32).to(device)
beta_bars=torch.cumprod(beta,dim=0).to(device)
alpha_bars_shifted=torch.cat((torch.tensor([1],dtype=torch.float32).to(device),alpha_bars[:-1]),dim=0)



def noise_scheduler(x_0):
  x_0=x_0.to(torch.float32).to(device)
  batch_size,color_channels,height,width=x_0.shape
    
  t=torch.randint(1,1000,(batch_size,1))
  epsilon=torch.randn((batch_size,color_channels,height,width),dtype=torch.float32).to(device)
  alpha_t_bars=alpha_bars[t].to(device)
  # print(alpha_t_bars.shape)
  alpha_t_bars=alpha_t_bars[...,None,None]
  # print(alpha_t_bars.shape)
  x_t=torch.sqrt(alpha_t_bars)*x_0+(torch.sqrt(1-alpha_t_bars)*epsilon)
  # print(x_t.dtype)
  # print(epsilon.dtype)
  # print(x_0.dtype)
  # x_t=x_t.squeeze(0)
  return x_t,epsilon,t
