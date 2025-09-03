import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
# ===================== 追加: 共通ユーティリティ =====================
def resize_to(img_2d, new_w, new_h, resample=Image.LANCZOS, dtype=np.float32):
    """2D float配列をPILでリサイズ（Lanczos推奨）。"""
    # PILは(H,W)を(F)モードで扱える。W,Hの順で指定に注意
    pil = Image.fromarray(np.asarray(img_2d, dtype=np.float32), mode='F')
    pil = pil.resize((new_w, new_h), resample=resample)
    arr = np.asarray(pil, dtype=dtype)
    return arr
def wiener_like_filter(G, H, k=None):
    """
    F_hat = G * H.conj() / (|H|^2 + k)  （k=Noneなら位相共役フィルタ）
    """
    if k is None:
        H_inv = np.conj(H)
    else:
        H_mag2 = (np.abs(H) ** 2)
        H_inv = np.conj(H) / (H_mag2 + k)
    return G * H_inv
# ===================== 元の関数（小改良） =====================
# 符号化画像の生成(式(2),(3))
def generate_code_images(gray_image, phases, beta_prime, X, Y):
    print("畳み込み処理...")
    coded_images = []
    for i, phi in enumerate(phases):
        # PSFをFZAの幾何学的影で近似(式(2))
        h_FZA = 0.5 * (1 + np.cos(beta_prime * (X**2 + Y**2) - phi))
        visualize_psf(h_FZA,i)
        # グレースケールで畳み込み(式(3))
        coded_image = signal.fftconvolve(gray_image, h_FZA, mode='same')
        coded_images.append(coded_image.astype(np.float32))
    print("畳み込み完了")
    return coded_images
def add_awgn(signal_img, snr_db):
    signal_power = float(np.var(signal_img))
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-12)
    noise = np.random.normal(0.0, np.sqrt(noise_power), signal_img.shape).astype(np.float32)
    return (signal_img + noise).astype(np.float32)
def quantize_to_12bit(signal_img):
    # 0-1に正規化 → 12bit量子化 → 元スケールへ戻す
    sig_min, sig_max = float(np.min(signal_img)), float(np.max(signal_img))
    rng = max(sig_max - sig_min, 1e-12)
    norm_signal = (signal_img - sig_min) / rng
    quantized = np.round(norm_signal * 4095.0)
    dequantized = (quantized / 4095.0).astype(np.float32)
    return (dequantized * rng + sig_min).astype(np.float32)
# フリンジスキャン(式(6))
def synthesize_fringe_scan(coded_images):
    print("FS...")
    g_0, g_pi2, g_pi, g_3pi2 = coded_images
    j = 1j
    g_FS = -(g_pi2 - g_3pi2) + j * (g_0 - g_pi)
    print("FS完了")
    return g_FS.astype(np.complex64)
# ===================== センサ解像度での復元 =====================
def reconstruct_image_with_psf(g_FS_sensor, h_geo_sensor, wiener_k=None):
    """
    g_FS_sensor: センサ解像度の複素画像
    h_geo_sensor: センサ解像度にリサイズ済みの h_geo（複素）
    wiener_k: Noneなら位相共役, 値を与えるとWiener風に安定化
    """
    # 注意: h_geo は空間領域のインパルス応答 → ifftshiftしてからFFT
    H = np.fft.fft2(np.fft.ifftshift(h_geo_sensor))
    G = np.fft.fft2(g_FS_sensor)
    
    # === Hの値を検査するコードを追加 ===
    H_abs = np.abs(H)
    print(f"Hの最小絶対値: {np.min(H_abs)}")
    print(f"Hのゼロに近い値の数（例: 1e-10未満）: {np.sum(H_abs < 1e-10)}")
    
    # === Hの絶対値の分布を可視化 ===
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log(H_abs), cmap='viridis')
    plt.title("Frequency Spectrum of H (log scale)")
    plt.colorbar(label='log(|H|)')
    plt.show()
    # ================================

    # F_hat = wiener_like_filter(G, H, k=wiener_k)
    F_hat=G*np.conj(H)
    f_geo = np.fft.ifft2(F_hat)
    return np.abs(f_geo).astype(np.float32)
# -----------------------------------------------------------------------
# 可視化（そのまま）
def visualize_psf(h_FZA,i):
    # PSFを画像として表示してエイリアシングを確認する
    plt.figure(figsize=(6, 6))
    plt.imshow(h_FZA, cmap='gray')
    plt.title(f"PSF (h_FZA) for Aliasing Check (φ={i*90}°)")
    plt.colorbar()
    plt.show()
def visualize_coded_images(original_img, coded_images, phase_labels):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Grayscale Image")
    for i, img in enumerate(coded_images):
        axes[i+1].imshow(img, cmap='gray')
        axes[i+1].set_title(f"Coded ({phase_labels[i]})")
    plt.tight_layout()
    plt.show()
def visualize_synthesis_result(g_FS):
    magnitude = np.abs(g_FS)
    phase = np.angle(g_FS)
    phase[phase < 0] += 2 * np.pi
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fringe Scan Synthesis Result", fontsize=16)
    im1 = axes[0].imshow(magnitude, cmap='gray')
    axes[0].set_title("Amplitude")
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(phase, cmap='hsv', vmin=0, vmax=2*np.pi)
    axes[1].set_title("Phase")
    cbar = fig.colorbar(im2, ax=axes[1])
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    plt.tight_layout()
    plt.show()
def visualize_final_result(original_img, reconstructed_image_raw, title="Reconstructed (Sensor size)"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Final Comparison", fontsize=16)
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Image (SLM size)")
    axes[1].imshow(reconstructed_image_raw, cmap='gray')
    axes[1].set_title(title)
    plt.tight_layout()
    plt.show()
# -----------------------------------------------------------------------
if __name__ == '__main__':
    # ==== SLM仕様（従来通り）====
    slm_resolution_x, slm_resolution_y = 1024, 768
    slm_pixel_pitch = 36e-6  # 36 µm
    slm_width  = slm_resolution_x * slm_pixel_pitch
    slm_height = slm_resolution_y * slm_pixel_pitch
    x = np.linspace(-slm_width / 2,  slm_width / 2,  slm_resolution_x, dtype=np.float64)
    y = np.linspace(-slm_height / 2, slm_height / 2, slm_resolution_y, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    # ==== センサ仕様（新規）====
    sensor_resolution_x, sensor_resolution_y = 4104, 3006
    sensor_pixel_pitch = 3.45e-6  # 3.45 µm （今回は座標に直接は使わない）
    # ==== パラメータ ====
    beta = 0.7e6
    d = 0.5e-2
    z = 2.5e-1
    M = d / z
    beta_prime = beta / ((M + 1) ** 2)
    snr_db = 30.0
    phases = [0, np.pi/2, np.pi, 3*np.pi/2]
    phase_labels = ["φ=0", "φ=π/2", "φ=π", "φ=3π/2"]
    # ==== 入力画像（SLM解像度に合わせて読み込み）====
    image_path = r"C:\Users\ko11s\OneDrive\Desktop\research\b\img\subject.jpg"
    input_image_uint8 = np.array(
        Image.open(image_path).convert("L").resize((slm_resolution_x, slm_resolution_y), resample=Image.LANCZOS)
    )
    input_image = input_image_uint8.astype(np.float32)
    # ==== 1) 符号化画像（SLMサイズ）====
    coded_images_slm = generate_code_images(input_image, phases, beta_prime, X, Y)
    # ==== 2) 4枚をセンサ解像度へリサイズ（Lanczos推奨）====
    coded_images_sensor = [
        resize_to(img, sensor_resolution_x, sensor_resolution_y, resample=Image.LANCZOS, dtype=np.float32)
        for img in coded_images_slm
    ]
    # ==== 3) AWGN & 12bit量子化（センササイズで実行）====
    noisy_quantized_images_sensor = []
    for img in coded_images_sensor:
        img_noisy = add_awgn(img, snr_db)         # AWGN（例: 30 dB）
        img_q12  = quantize_to_12bit(img_noisy)   # 12-bit ADCを模擬
        noisy_quantized_images_sensor.append(img_q12)
    # ==== 4) FS（センササイズ）====
    g_FS_sensor = synthesize_fringe_scan(noisy_quantized_images_sensor)
    # ==== 5) 逆フィルタ用 h_geo を作成してからセンサ解像度にリサイズ ====
    #     ※サンプリング一貫性のため、SLMグリッドで作成→センサ解像度へ補間
    h_geo_slm = np.exp(1j * beta_prime * (X**2 + Y**2)).astype(np.complex64)
    h_geo_sensor_real = resize_to(np.real(h_geo_slm), sensor_resolution_x, sensor_resolution_y, resample=Image.LANCZOS, dtype=np.float32)
    h_geo_sensor_imag = resize_to(np.imag(h_geo_slm), sensor_resolution_x, sensor_resolution_y, resample=Image.LANCZOS, dtype=np.float32)
    h_geo_sensor = (h_geo_sensor_real + 1j * h_geo_sensor_imag).astype(np.complex64)
    # ==== 6) 復元（位相共役 or 安定化Wiener）====
    # 位相共役（従来同等）
    reconstructed_pc = reconstruct_image_with_psf(g_FS_sensor, h_geo_sensor, wiener_k=None)
    # 参考: 安定化したい場合（リサイズ後の高周波で騒がしくなるとき）
    # snr_linear = 10 ** (snr_db / 10.0)
    # k = (1.0 / snr_linear) * np.mean(np.abs(np.fft.fft2(np.fft.ifftshift(h_geo_sensor)))**2)
    # reconstructed_wiener = reconstruct_image_with_psf(g_FS_sensor, h_geo_sensor, wiener_k=k)
    # ==== 可視化（必要に応じて）====
    # SLMサイズでの符号化画像の確認
    # visualize_coded_images(input_image_uint8, coded_images_slm, phase_labels)
    # センササイズでのFS結果
    visualize_synthesis_result(g_FS_sensor)
    # 最終比較（左: 元画像(SLMサイズ) / 右: 復元(センササイズ)）
    visualize_final_result(input_image_uint8, reconstructed_pc, title="Reconstructed (Sensor size, phase-conjugate)")
