import os
import torch
from torchvision.utils import save_image


@torch.no_grad()
def sample(model, img_size, alphas, alphas_cumprod, betas, T, device, batch_size=64):
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)  # 순수 노이즈로부터 시작
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

        x = (1 / torch.sqrt(alphas[t])) * (x - (betas[t] / torch.sqrt(1 -
                                                                      alphas_cumprod[t])) * noise_pred) + torch.sqrt(beta_t) * noise
    return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02

    betas = torch.linspace(beta_start, beta_end, T).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    model = torch.load("ddpm.pth")

    sample_images = sample(model, 32, alphas, alphas_cumprod, betas, T, device, batch_size=16)
    os.makedirs('generated_images', exist_ok=True)
    sample_images = (sample_images + 1) / 2  # [-1,1] -> [0,1]
    for idx, img in enumerate(sample_images):
        save_image(img, f'generated_images/{idx}.png')


if __name__ == '__main__':
    main()
