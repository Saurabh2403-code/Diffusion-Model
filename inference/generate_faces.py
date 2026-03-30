VAE_model.eval()
net.eval()
frames=[]
x_t = torch.randn(16, 4, 16, 16, dtype=torch.float32).to(device)

with torch.inference_mode():
    for i in tqdm(range(1000)):
        t = 999 - i
        t_normalized = torch.full((16,), t/1000, dtype=torch.float32).to(device)
        
        predicted_noise = net(x_t, t_normalized)
        
        term_1 = 1/torch.sqrt(alpha[t])
        term_2 = x_t - (beta[t]/torch.sqrt(1-alpha_bars[t]))*predicted_noise
        x_t_1 = term_1 * term_2
        
        sigma_t = torch.sqrt(((1-alpha_bars_shifted[t])/(1-alpha_bars[t]))*beta[t])
        if t > 0:
            x_t_1 = x_t_1 + sigma_t * torch.randn(16, 4, 16, 16, dtype=torch.float32).to(device)
        x_t = x_t_1

    # Decode latents to images using VAE decoder
    x_t=(x_t*std_global)+mean_global
    generated_images = VAE_model.decoder(x_t)

# Plot
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for idx, ax in enumerate(axes.flatten()):
    image = generated_images[idx].cpu()
    image = (image + 1) / 2
    image = image.permute(1, 2, 0)
    image = image.numpy().clip(0, 1)
    ax.imshow(image)
    ax.axis('off')
plt.tight_layout()
plt.show()
