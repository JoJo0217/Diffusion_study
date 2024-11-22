import os
import torch
from torchvision.utils import save_image


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02

    betas = torch.linspace(beta_start, beta_end, T).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    model = torch.load("ddpm.pth")

    sample_images = sample(model, 28, alphas, alphas_cumprod, betas, T, device, batch_size=8)
    os.makedirs('generated_images', exist_ok=True)
    sample_images = (sample_images + 1) / 2  # [-1,1] -> [0,1]
    for time_step, images in enumerate(reversed(sample_images)):
        for idx, img in enumerate(images):
            save_image(img, f'generated_images/{time_step*25}_{idx}.png')


if __name__ == '__main__':
    main()
