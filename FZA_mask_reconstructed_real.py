import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

# リサイズ関数
def resize_to(img_2d, new_w, new_h, resample=Image.LANCZOS, dtype=np.float32):
    # PILは(H,W)を(F)モードで扱える。W,Hの順で指定に注意
    pil = Image.fromarray(np.asarray(img_2d, dtype=np.float32), mode='F')
    pil = pil.resize((new_w, new_h), resample=resample)
    arr = np.asarray(pil, dtype=dtype)
    return arr

# 符号化画像の生成(式(2),(3))
def generate_code_images(gray_image,phases,beta_prime,X,Y):
    # 4つの位相を持つPSFで畳み込みを行い、４枚の符号化画像を生成
    print("畳み込み処理...")
    # 符号化画像を格納するリスト
    coded_images=[]

    i=0

    for phi in phases:
        # PSFをFZAの幾何学的影で近似(式(2))
        h_FZA=0.5*(1+np.cos(beta_prime*(X**2+Y**2)-phi))

        # FZAのエイリアシングを確認
        visualize_psf(h_FZA,i)

        # グレースケールで畳み込み式((3))
        coded_image=signal.fftconvolve(gray_image, h_FZA, mode='same')
        coded_images.append(coded_image)
        i+=1

    print("畳み込み完了")
    return coded_images

def add_awgn(signal, snr_db):
    # 信号パワー
    signal_power = np.var(signal)
    # 目標SNRから雑音パワーを算出
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power / snr_linear
    
    # ガウス雑音生成
    # 引数(平均、標準偏差、画像データのサイズ分のノイズ)
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    return signal + noise

def quantize_to_12bit(signal):
    # 最小値0, 最大値1 に正規化
    sig_min, sig_max = np.min(signal), np.max(signal)
    norm_signal = (signal - sig_min) / (sig_max - sig_min + 1e-12)
    
    # 12bitに量子化 (0〜4095)
    quantized = np.round(norm_signal * 4095)
    
    # 0〜1に戻す
    dequantized = quantized / 4095.0

    # 元スケールに戻す(逆正規化)
    return dequantized * (sig_max - sig_min) + sig_min


# フリンジスキャン(式(6))
def synthesize_fringe_scan(coded_images):
    # ４枚の複素数画像を１枚に合成
    print("FS...")
    g_0,g_pi_half,g_pi,g_3pi_half=coded_images[0],coded_images[1],coded_images[2],coded_images[3]
    j=1j

    # グレースケールなので一回のFS
    g_FS=-(g_pi_half-g_3pi_half)+j*(g_0-g_pi)

    print("FS完了")
    return g_FS


# 逆フィルタをセンササイズへリサイズ
def resize_inv_filter(X_slm,Y_slm,beta_prime):
    # 単一複素数伝達関数の作成
    # h_geoを計算(式(7))
    h_geo_slm=np.exp(1j*beta_prime*(X_slm**2+Y_slm**2)).astype(np.complex64)
    
    # リサイズ
    h_geo_sensor_real=resize_to(np.real(h_geo_slm),sensor_resolution_x,sensor_resolution_y,resample=Image.BILINEAR, dtype=np.float32)
    h_geo_sensor_imag=resize_to(np.imag(h_geo_slm),sensor_resolution_x,sensor_resolution_y,resample=Image.BILINEAR, dtype=np.float32)
    h_geo_sensor=(h_geo_sensor_real+1j*h_geo_sensor_imag).astype(np.complex64)
    
    return h_geo_sensor


# 画像再構成(式(8),(9))
def reconstruct_image(g_FS_sensor,h_geo_sensor):
    j=1j

    # 複素数画像から逆畳み込みにより画像再構成
    h_geo_sensor=np.fft.fft2(np.fft.ifftshift(h_geo_sensor))
    # 逆フィルタの作成
    H_geo_inv_sensor=np.conj(h_geo_sensor)

    # 画像再構成(式(9))
    G_FS_sensor=np.fft.fft2(g_FS_sensor)

    # 周波数領域で逆フィルタを適用
    F_reconstructed_sensor=G_FS_sensor*H_geo_inv_sensor

    # 逆フーリエ変換で空間領域へ戻す
    f_geo_sensor=np.fft.ifft2(F_reconstructed_sensor)

    # 絶対値を取る
    reconstructed_image_raw=np.abs(f_geo_sensor)

    return reconstructed_image_raw


# -----------------------------------------------------------------------


# 可視化用関数

def visualize_psf(h_FZA,i):
    # PSFを画像として表示してエイリアシングを確認する
    plt.figure(figsize=(6, 6))
    plt.imshow(h_FZA, cmap='gray')
    plt.title(f"PSF (h_FZA) for Aliasing Check (φ={i*90}°)")
    plt.colorbar()
    plt.savefig(f'img/FZA/FZA_phi_{i*90}_real.png')
    plt.show()


def visualize_coded_images(original_img,coded_images,phase_labels):
    # ４枚の符号化画像を可視化
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Grayscale Image")
    for i, img in enumerate(coded_images):
        axes[i+1].imshow(img, cmap='gray')
        axes[i+1].set_title(f"Coded Image ({phase_labels[i]})")
    plt.tight_layout()
    plt.savefig('img/coded_image/coded_images_real.png')
    plt.show()


def visualize_synthesis_result(g_FS):
    # 複素数画像の振幅と位相を可視化
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
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('img/FS/fringe_scan_synthesis_real.png')
    plt.show()


def visualize_final_result(original_img, reconstructed_image_raw):
    # 最終的な再構成結果をオリジナル画像と比較して表示
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Final Comparison", fontsize=16)
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[1].imshow(reconstructed_image_raw, cmap='gray')
    axes[1].set_title("Reconstructed Image (Conventional)")
    plt.tight_layout()
    plt.savefig('img/final_image/final_reconstruction_real.png')
    plt.show()


# -----------------------------------------------------------------------



if __name__ == '__main__':
    # SLM
    # SLMの仕様
    slm_resolution_x,slm_resolution_y=1024,768
    slm_pixel_pitch=36e-6  # 36um(1pxのサイズ)

    # SLMの物理的なサイズを計算
    slm_width=slm_resolution_x*slm_pixel_pitch
    slm_height=slm_resolution_y*slm_pixel_pitch

    # SLMの座標グリッドを生成
    x_slm=np.linspace(-slm_width/2,slm_width/2,slm_resolution_x)
    y_slm=np.linspace(-slm_height/2,slm_height/2,slm_resolution_y)
    X_slm,Y_slm=np.meshgrid(x_slm,y_slm)
    
    # Sensor
    # sensorの仕様
    sensor_resolution_x,sensor_resolution_y=4104,3006
    sensor_pixel_pitch=3.45e-6  # 3.45µm

    # パラメータの設定
    beta=0.6e6
    d=0.5e-2
    z=2.5e-1
    M=d/z
    beta_prime=beta/((M+1)**2)
    snr_db=30.0
    phases=[0,np.pi/2,np.pi,3*np.pi/2]
    phase_labels=["φ=0", "φ=π/2", "φ=π", "φ=3π/2"]

    # 被写体画像のパス
    image_path="C:\\Users\\ko11s\\OneDrive\\Desktop\\research\\b\\img\\subject.jpg" 

    # 画像をSLMの解像度(1024x768)にリサイズして読み込み
    input_image_uint8 = np.array(
        Image.open(image_path)
        .convert("L")  # グレースケールに変換
        .resize((slm_resolution_x, slm_resolution_y)) # SLMの解像度にリサイズ
    )
    input_image = input_image_uint8.astype(np.float64)


    # 処理を順に実行
    # 符号化画像を生成(SLMサイズ)
    coded_images_slm=generate_code_images(input_image,phases,beta_prime,X_slm,Y_slm)

    # 4枚をセンサ解像度へリサイズ
    coded_images_sensor = [
        resize_to(img, sensor_resolution_x, sensor_resolution_y, resample=Image.BILINEAR, dtype=np.float32)
        for img in coded_images_slm
    ]

    # AWGN & 量子化を各符号化画像に適用
    noisy_quantized_images_sensor = []
    for img in coded_images_sensor:
        img_noisy = add_awgn(img, snr_db) # 30dB AWGN
        img_quantized = quantize_to_12bit(img_noisy)
        noisy_quantized_images_sensor.append(img_quantized)

    # フリンジスキャンを実行
    g_FS_sensor=synthesize_fringe_scan(noisy_quantized_images_sensor)

    # フィルタを作成
    h_geo_sensor=resize_inv_filter(X_slm,Y_slm,beta_prime)
    
    # 画像を再生成
    reconstructed_image_raw=reconstruct_image(g_FS_sensor,h_geo_sensor)

    # 結果の可視化
    visualize_coded_images(input_image_uint8, coded_images_sensor, phase_labels)
    visualize_synthesis_result(g_FS_sensor)
    visualize_final_result(input_image_uint8, reconstructed_image_raw)