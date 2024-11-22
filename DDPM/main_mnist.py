import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from model import MyUNet
from sampling import sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 10
batch_size = 128
lr = 1e-3


model = MyUNet().to(device)

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

        if i % 100 == 0:
            print(f"Epoch [{epoch}], Step [{i}], Loss: {noise_loss.item()}")

torch.save(model, "ddpm.pth")


@torch.no_grad()
def sample(model, img_size, alphas, alphas_cumprod, betas, T, device, batch_size=64):
    model.eval()
    x = torch.randn(batch_size, 1, img_size, img_size).to(device)  # 순수 노이즈로부터 시작
    noise_to_x = [x]
    for t in reversed(range(T)):
        # 현재 타임스텝 t
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # 노이즈 예측합니다.
        noise_pred = model(x, t_tensor)

        beta_t = betas[t].to(device)

        # 이전 x를 계산합니다.
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alphas[t])) * (x - ((1 - alphas[t]) / torch.sqrt(1 -
                                                                             alphas_cumprod[t])) * noise_pred) + torch.sqrt(beta_t) * noise
        if t == 0 or t == 250 or t == 500 or t == 750:
            noise_to_x.append(x)
    noise_to_x = torch.stack(noise_to_x, dim=0)
    return noise_to_x


# test
# time, batch, channel, height, width
sample_images = sample(model, 28, alphas, alphas_cumprod, betas, T, device, batch_size=8)
os.makedirs('generated_images', exist_ok=True)
sample_images = (sample_images + 1) / 2  # [-1,1] -> [0,1]
for time_step, images in enumerate(reversed(sample_images)):
    for idx, img in enumerate(images):
        save_image(img, f'generated_images/{time_step*25}_{idx}.png')
