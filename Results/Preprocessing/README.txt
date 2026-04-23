Filter Performance Summary:

The preprocessing stage evaluates multiple denoising filters using the following metrics: SNR, PSNR, CNR, SSIM, RMSE, EPI, and Noise Standard Deviation. All metrics were normalized (with RMSE and noise inverted so that higher values indicate better performance), and a composite score was computed to compare overall filter quality.

Ranking based on composite score:

Bilateral (~0.84) > Gaussian (~0.73) > Wiener (~0.72) > Median (~0.69) > Butterworth (~0.11)

The bilateral filter achieved the best overall balance between noise reduction and structural preservation.

Key Observations

Bilateral Filter:
Achieved the highest composite score. It maintained strong SNR (~14.25 dB) and CNR (~1.12) while preserving structural details (SSIM ~0.85, EPI ~0.84). The edge-preserving nature of the bilateral filter allows effective noise reduction without blurring important boundaries.

Median Filter:
Performed best in PSNR (~39.16 dB) and SSIM (~0.95) with the lowest RMSE (~0.0117), indicating strong overall similarity and effective impulse noise removal. However, it reduced contrast (lower CNR ~1.01), leading to loss of fine details in certain regions.

Wiener Filter:
Provided strong edge preservation (EPI ~0.93) and low error (RMSE ~0.0175). It achieved good PSNR (~35.46 dB) but slightly lower SNR (~13.89 dB), resulting in overall performance just below Gaussian.

Gaussian Filter:
Produced consistent smoothing with high SNR (~14.22 dB) and reasonable noise reduction. However, it introduced noticeable blurring, reducing structural sharpness (lower SSIM compared to median and bilateral).

Butterworth Filter:
Performed worst across most metrics, with low SSIM (~0.61), high RMSE (~0.108), and weak SNR/PSNR values. It was not suitable for this application.

Final Recommendation

The bilateral filter is selected as the optimal preprocessing method. It provides the best trade-off between noise reduction and edge preservation, which is critical for maintaining nodule boundaries in subsequent segmentation and classification stages.