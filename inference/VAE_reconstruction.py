VAE_model.eval()
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
with torch.inference_mode():
    for idx,ax in enumerate(axes.flatten()):
        input_image=train_data[idx][0].unsqueeze(0)
        # print(f'input_image {input_image.shape}')
        input_image=input_image.to(device)
        log_var,mean,predicted_image=VAE_model(input_image)
        # print(f'log_var {log_var.shape}')
        # print(f'mean {mean.shape}')
        # print(f'predicted_image {predicted_image.shape}')
        predicted_image=predicted_image.permute(0,2,3,1)
        # print(f'predicted_image {predicted_image.shape}')
        predicted_image=(predicted_image+1)/2
        predicted_image=predicted_image.clip(0,1)
        predicted_image=predicted_image.to('cpu')
        predicted_image=predicted_image.detach().numpy()
        ax.imshow(predicted_image.squeeze(0))
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
