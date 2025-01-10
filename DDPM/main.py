import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from model import MyUNet, DiT
from sampling import sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 10
batch_size = 128
lr = 1e-3
model_dir = 'ddpm.pth'
img_save_dir = 'generated_images'


# model = MyUNet().to(device)
model = DiT().to(device)


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
    transforms.Normalize((0.5,), (0.5,))  # 데이터를 [-1, 1] 범위로 정규화
])


dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in tqdm(range(epochs)):
    total_loss = 0
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device)
        batch_size = x.size(0)

        t = torch.randint(0, T, (batch_size,)).to(device)

        noise = torch.randn_like(x).to(device)

        # 이미지에 노이즈 추가
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(x.shape[0], 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(
            x.shape[0], 1, 1, 1)
        # reparametrization trick
        noise_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

        noise_pred = model(noise_x, t)

        noise_loss = criterion(noise_pred, noise)
        optimizer.zero_grad()
        noise_loss.backward()
        optimizer.step()
        total_loss += noise_loss.item()

    print(f"Epoch [{epoch}], Loss: {total_loss/len(train_loader)}")

torch.save(model, model_dir)


# test
# time, batch, channel, height, width
sample_images = samples = sample(model, 28, alphas, alphas_cumprod, betas, T, device, batch_size=8)

os.makedirs(img_save_dir, exist_ok=True)
sample_images = sample_images.detach().cpu().numpy()
sample_images = (sample_images + 1) / 2  # [-1,1] -> [0,1]

for idx, img in enumerate(sample_images):
    # pil_img = Image.fromarray(img)
    img = np.squeeze(img)
    file_path = os.path.join(img_save_dir, f"{idx}.png")
    # pil_img.save(file_path)
    plt.imsave(file_path, img, cmap='gray')
