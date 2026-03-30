import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader as DataLoader

import kagglehub
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

print("Path to dataset files:", '/kaggle/working/CelebA_Dataset')

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = datasets.ImageFolder(
    root='/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba/',
    transform=transform
)

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=128,
    shuffle=True,
    num_workers=2
)
