
"""
section3_image_recovery.py

Generates:
1) Denoising and inpainting results for Dataset 1 (scikit-image images)
2) Report-ready figures:
   - section3_outputs/figures/camera_denoising.png
   - section3_outputs/figures/camera_inpainting.png
   - section3_outputs/figures/psnr_vs_sampling.png
3) Quantitative metrics CSV:
   - section3_outputs/results_metrics.csv

Dependencies:
    pip install numpy matplotlib scikit-image PyWavelets pandas
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

from skimage import data, color, img_as_float
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.data import shepp_logan_phantom


ROOT = "section3_outputs"
FIG_DIR = os.path.join(ROOT, "figures")
IMG_DIR = os.path.join(ROOT, "images")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


def preprocess_image(img, size=128):
    img = img_as_float(img)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = resize(img, (size, size), anti_aliasing=True)
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img


def load_dataset1(size=128):
    return {
        "camera": preprocess_image(data.camera(), size=size),
        "astronaut_gray": preprocess_image(data.astronaut(), size=size),
        "coins": preprocess_image(data.coins(), size=size),
        "shepp_logan": preprocess_image(shepp_logan_phantom(), size=size),
    }


def add_gaussian_noise(img, sigma=0.05, seed=42):
    rng = np.random.default_rng(seed)
    noisy = img + sigma * rng.standard_normal(img.shape)
    return np.clip(noisy, 0.0, 1.0)


def random_pixel_mask(img, keep_ratio=0.5, seed=42):
    rng = np.random.default_rng(seed)
    mask = (rng.random(img.shape) < keep_ratio).astype(np.float32)
    return img * mask, mask


def compute_metrics(gt, img):
    rmse = np.sqrt(np.mean((gt - img) ** 2))
    psnr = peak_signal_noise_ratio(gt, img, data_range=1.0)
    ssim = structural_similarity(gt, img, data_range=1.0)
    return {"RMSE": float(rmse), "PSNR": float(psnr), "SSIM": float(ssim)}


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


class WaveletTransform2D:
    def __init__(self, image_shape, wavelet="db4", level=3, mode="periodization"):
        self.image_shape = tuple(image_shape)
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        zero = np.zeros(self.image_shape, dtype=np.float32)
        coeffs = pywt.wavedecn(zero, wavelet=self.wavelet, level=self.level, mode=self.mode)
        arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        self.coeff_shape = arr.shape
        self.coeff_slices = coeff_slices
        self.n_coeff = arr.size

    def image_to_coeffs(self, img):
        coeffs = pywt.wavedecn(img, wavelet=self.wavelet, level=self.level, mode=self.mode)
        arr, _ = pywt.coeffs_to_array(coeffs)
        return arr.astype(np.float32)

    def coeffs_to_image(self, arr):
        coeffs = pywt.array_to_coeffs(arr, self.coeff_slices, output_format="wavedecn")
        img = pywt.waverecn(coeffs, wavelet=self.wavelet, mode=self.mode)
        img = img[:self.image_shape[0], :self.image_shape[1]]
        return img.astype(np.float32)


def wavelet_denoise(y, wavelet="db4", level=3, tau=0.04):
    wt = WaveletTransform2D(y.shape, wavelet=wavelet, level=level)
    coeffs = wt.image_to_coeffs(y)
    coeffs_denoised = soft_threshold(coeffs, tau)
    x_hat = wt.coeffs_to_image(coeffs_denoised)
    return np.clip(x_hat, 0.0, 1.0)


class InpaintingWaveletOperator:
    def __init__(self, image_shape, mask, wavelet="db4", level=3, mode="periodization"):
        self.mask = mask.astype(np.float32)
        self.wt = WaveletTransform2D(image_shape, wavelet=wavelet, level=level, mode=mode)

    @property
    def n_coeff(self):
        return self.wt.n_coeff

    def forward(self, alpha):
        arr = alpha.reshape(self.wt.coeff_shape)
        img = self.wt.coeffs_to_image(arr)
        return (self.mask * img).ravel()

    def adjoint(self, residual_vec):
        residual = residual_vec.reshape(self.mask.shape)
        coeffs = self.wt.image_to_coeffs(self.mask * residual)
        return coeffs.ravel()

    def coeffs_to_image(self, alpha):
        return self.wt.coeffs_to_image(alpha.reshape(self.wt.coeff_shape))


def estimate_lipschitz(op, n_iter=20, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(op.n_coeff).astype(np.float32)
    x /= (np.linalg.norm(x) + 1e-12)
    for _ in range(n_iter):
        x = op.adjoint(op.forward(x))
        x /= (np.linalg.norm(x) + 1e-12)
    Ax = op.forward(x)
    AtAx = op.adjoint(Ax)
    L = float(np.dot(x, AtAx))
    return max(L, 1e-6)


def fista_l1(op, y_vec, tau, max_iter=400, tol=1e-5, L=None):
    n = op.n_coeff
    if L is None:
        L = estimate_lipschitz(op)
    x = np.zeros(n, dtype=np.float32)
    z = x.copy()
    t = 1.0
    obj_hist = []

    for _ in range(max_iter):
        grad = op.adjoint(op.forward(z) - y_vec)
        x_new = soft_threshold(z - grad / L, tau / L)

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)

        resid = op.forward(x_new) - y_vec
        obj = 0.5 * np.dot(resid, resid) + tau * np.sum(np.abs(x_new))
        obj_hist.append(obj)

        rel_change = np.linalg.norm(x_new - x) / max(1.0, np.linalg.norm(x))
        x = x_new
        t = t_new
        if rel_change < tol:
            break

    return x, np.array(obj_hist), L


def wavelet_inpaint(masked_img, mask, wavelet="db4", level=3, tau=0.008, max_iter=400):
    op = InpaintingWaveletOperator(masked_img.shape, mask, wavelet=wavelet, level=level)
    y_vec = masked_img.ravel()
    alpha_hat, obj_hist, L = fista_l1(op, y_vec, tau=tau, max_iter=max_iter, tol=1e-5)
    x_hat = op.coeffs_to_image(alpha_hat)
    return np.clip(x_hat, 0.0, 1.0), alpha_hat, obj_hist, L


def save_png(img, path):
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def make_camera_denoising_figure(clean, denoise_dict, path):
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    sigmas = [0.01, 0.05, 0.10]

    axs[0, 0].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axs[0, 0].set_title("Clean")
    axs[0, 0].axis("off")
    axs[1, 0].axis("off")

    for j, sigma in enumerate(sigmas, start=1):
        axs[0, j].imshow(denoise_dict[sigma]["noisy"], cmap="gray", vmin=0, vmax=1)
        axs[0, j].set_title(f"Noisy ($\\sigma={sigma}$)")
        axs[0, j].axis("off")

        axs[1, j].imshow(denoise_dict[sigma]["recon"], cmap="gray", vmin=0, vmax=1)
        axs[1, j].set_title("Reconstructed")
        axs[1, j].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def make_camera_inpainting_figure(clean, inp_dict, path):
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    keeps = [0.3, 0.5, 0.7]

    axs[0, 0].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axs[0, 0].set_title("Clean")
    axs[0, 0].axis("off")
    axs[1, 0].axis("off")

    for j, keep in enumerate(keeps, start=1):
        axs[0, j].imshow(inp_dict[keep]["masked"], cmap="gray", vmin=0, vmax=1)
        axs[0, j].set_title(f"Masked ({int(keep*100)}%)")
        axs[0, j].axis("off")

        axs[1, j].imshow(inp_dict[keep]["recon"], cmap="gray", vmin=0, vmax=1)
        axs[1, j].set_title("Reconstructed")
        axs[1, j].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def make_psnr_sampling_plot(df, path):
    sub = df[(df["image"] == "camera") & (df["task"] == "inpainting")].sort_values("parameter")
    plt.figure(figsize=(6, 4))
    plt.plot(sub["parameter"], sub["recon_PSNR"], marker="o")
    plt.xlabel("Sampling ratio")
    plt.ylabel("Reconstruction PSNR (dB)")
    plt.title("Camera: PSNR vs Sampling Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    images = load_dataset1(size=128)
    rows = []

    noise_levels = [0.01, 0.05, 0.10]
    keep_ratios = [0.3, 0.5, 0.7]

    denoise_tau = {0.01: 0.008, 0.05: 0.03, 0.10: 0.06}
    inpaint_tau = {0.3: 0.010, 0.5: 0.008, 0.7: 0.005}

    camera_denoise = {}
    camera_inpaint = {}

    for name, clean in images.items():
        np.save(os.path.join(IMG_DIR, f"{name}.npy"), clean)
        save_png(clean, os.path.join(IMG_DIR, f"{name}.png"))

    for name, clean in images.items():
        for sigma in noise_levels:
            noisy = add_gaussian_noise(clean, sigma=sigma, seed=42)
            recon = wavelet_denoise(noisy, wavelet="db4", level=3, tau=denoise_tau[sigma])

            noisy_metrics = compute_metrics(clean, noisy)
            recon_metrics = compute_metrics(clean, recon)

            rows.append({
                "image": name,
                "task": "denoising",
                "parameter": sigma,
                "tau": denoise_tau[sigma],
                "noisy_RMSE": noisy_metrics["RMSE"],
                "noisy_PSNR": noisy_metrics["PSNR"],
                "noisy_SSIM": noisy_metrics["SSIM"],
                "recon_RMSE": recon_metrics["RMSE"],
                "recon_PSNR": recon_metrics["PSNR"],
                "recon_SSIM": recon_metrics["SSIM"],
            })

            if name == "camera":
                camera_denoise[sigma] = {"noisy": noisy, "recon": recon}

    for name, clean in images.items():
        for keep in keep_ratios:
            masked, mask = random_pixel_mask(clean, keep_ratio=keep, seed=42)
            recon, _, obj_hist, L = wavelet_inpaint(
                masked, mask, wavelet="db4", level=3, tau=inpaint_tau[keep], max_iter=400
            )

            masked_metrics = compute_metrics(clean, masked)
            recon_metrics = compute_metrics(clean, recon)

            rows.append({
                "image": name,
                "task": "inpainting",
                "parameter": keep,
                "tau": inpaint_tau[keep],
                "L": L,
                "masked_RMSE": masked_metrics["RMSE"],
                "masked_PSNR": masked_metrics["PSNR"],
                "masked_SSIM": masked_metrics["SSIM"],
                "recon_RMSE": recon_metrics["RMSE"],
                "recon_PSNR": recon_metrics["PSNR"],
                "recon_SSIM": recon_metrics["SSIM"],
            })

            if name == "camera":
                camera_inpaint[keep] = {"masked": masked, "recon": recon, "mask": mask, "obj_hist": obj_hist}

    df = pd.DataFrame(rows)
    csv_path = os.path.join(ROOT, "results_metrics.csv")
    df.to_csv(csv_path, index=False)

    make_camera_denoising_figure(images["camera"], camera_denoise, os.path.join(FIG_DIR, "camera_denoising.png"))
    make_camera_inpainting_figure(images["camera"], camera_inpaint, os.path.join(FIG_DIR, "camera_inpainting.png"))
    make_psnr_sampling_plot(df, os.path.join(FIG_DIR, "psnr_vs_sampling.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(camera_inpaint[0.5]["obj_hist"])
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("FISTA objective (Camera, 50% sampling)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "camera_inpainting_objective.png"), dpi=220, bbox_inches="tight")
    plt.close()

    print("Saved outputs under:", ROOT)
    print("Metrics CSV:", csv_path)

    cam_den = df[(df["image"] == "camera") & (df["task"] == "denoising")][["parameter", "recon_PSNR", "recon_SSIM"]]
    cam_inp = df[(df["image"] == "camera") & (df["task"] == "inpainting")][["parameter", "recon_PSNR", "recon_SSIM"]]

    print("\nRepresentative Camera results:\n")
    print("Denoising:")
    print(cam_den.to_string(index=False))
    print("\nInpainting:")
    print(cam_inp.to_string(index=False))


if __name__ == "__main__":
    main()
