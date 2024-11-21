import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from model import UNet
from sampling import sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 10
batch_size = 64
lr = 1e-3

model = UNet().to(device)

T = 1000
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in tqdm(range(epochs)):
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device)
        batch_size = x.size(0)

        t = torch.randint(0, T, (batch_size,)).to(device).long()

        noise = torch.randn_like(x).to(device)

        # 이미지에 노이즈 추가
        alphas_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        # reparametrization trick
        noise_x = alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

        noise_pred = model(noise_x, t)
        noise_loss = nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        noise_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}], Step [{i}], Loss: {noise_loss.item()}")

torch.save(model, "ddpm.pth")

# test
sample_images = sample(model, 32, alphas, alphas_cumprod, betas, T, device, batch_size=16)
os.makedirs('generated_images', exist_ok=True)
sample_images = (sample_images + 1) / 2  # [-1,1] -> [0,1]
for idx, img in enumerate(sample_images):
    save_image(img, f'generated_images/{idx}.png')
