import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

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

        # グレースケールで畳み込みを計算(式(3))
        coded_image=signal.fftconvolve(gray_image,h_FZA,mode='same')
        coded_images.append(coded_image)
        
        i+=1

    print("畳み込み完了")
    return coded_images

def add_awgn(signal,snr_db):
    # 信号パワー
    signal_power=np.var(signal)

    # 目標SNRから雑音パワーを算出
    snr_linear=10**(snr_db/10)
    noise_power=signal_power/snr_linear
    
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

    g_FS=-(g_pi_half-g_3pi_half)+j*(g_0-g_pi)

    print("FS完了")
    return g_FS


# 画像再構成(式(8),(9))
def reconstruct_image(g_FS,beta_prime,X,Y):
    j=1j
    # 複素数画像から逆畳み込みにより画像再構成
    
    # 単一複素数伝達関数の作成
    # h_geoを計算(式(7))
    h_geo=np.exp(j*beta_prime*(X**2+Y**2))
    # h_heoをフーリエ変換(画像中央を(0,0)に変換してから)
    H_geo=np.fft.fft2(np.fft.ifftshift(h_geo))
    # 逆フィルタの作成
    H_geo_inv=np.conj(H_geo)

    # 画像再構成(式(9))
    G_FS=np.fft.fft2(g_FS)

    # 周波数領域で逆フィルタを適用
    F_reconstructed=G_FS*H_geo_inv

    # 逆フーリエ変換で空間領域へ戻す
    f_geo=np.fft.ifft2(F_reconstructed)

    # 絶対値を取る
    reconstructed_image_raw=np.abs(f_geo)

    return reconstructed_image_raw


# -----------------------------------------------------------------------


# 可視化用関数

def visualize_psf(h_FZA,i):
    # PSFを画像として表示してエイリアシングを確認する
    plt.figure(figsize=(6, 6))
    plt.imshow(h_FZA, cmap='gray')
    plt.title(f"PSF (h_FZA) for Aliasing Check (φ={i*90}°)")
    plt.colorbar()
    plt.savefig(f'img/FZA/FZA_phi_{i}_real.png')
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
    fig.suptitle("compare", fontsize=16)
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[1].imshow(reconstructed_image_raw, cmap='gray')
    axes[1].set_title("Reconstructed Image (Conventional)")
    plt.tight_layout()
    plt.savefig('img/final_image/final_reconstruction_real.png')
    plt.show()


# -----------------------------------------------------------------------



if __name__ == '__main__':
    # パラメータの設定(m単位)
    # SLMの仕様
    slm_pixel_pitch=36e-6
    slm_resolution_x,slm_resolution_y=1024,768
    slm_width=slm_resolution_x*slm_pixel_pitch
    slm_height=slm_resolution_y*slm_pixel_pitch

    # SLMの座標グリッドを生成
    x=np.linspace(-slm_width/2,slm_width/2,slm_resolution_x)
    y=np.linspace(-slm_height/2,slm_height/2,slm_resolution_y)
    X,Y=np.meshgrid(x, y)

    beta=6.5e5

    # 実際のパラメータ
    d=5.0e-3
    z=2.5e-1
    
    M=d/z
    beta_prime=beta/((M+1)**2)
    print(beta_prime)    

    # ノイズパラメータ(db)
    snr_db=30
    # 位相の定義
    phases=[0,np.pi/2,np.pi,3*np.pi/2]
    phase_labels=["φ=0", "φ=π/2", "φ=π", "φ=3π/2"]

    # 画像へのpath
    image_path="C:\\Users\\ko11s\\OneDrive\\Desktop\\research\\b\\img\\subject.jpg" 

    # カラー画像を読み込み、グレースケールおよびfloat型に変換
    input_image_uint8 = np.array(
        Image.open(image_path)
        .convert("L")  # グレースケール(Luminance)に変換
        .resize((slm_resolution_x, slm_resolution_y)) # SLMの解像度にリサイズ
    )
    input_image=input_image_uint8.astype(np.float64)

    # 処理を順に実行
    # 符号化画像を生成
    coded_images=generate_code_images(input_image,phases,beta_prime,X,Y)

    # AWGN & 量子化を各符号化画像に適用
    noisy_quantized_images = []
    for img in coded_images:
        img_noisy = add_awgn(img, snr_db)  # 30dB AWGN
        img_quantized = quantize_to_12bit(img_noisy)
        noisy_quantized_images.append(img_quantized)

    # フリンジスキャンを実行
    g_FS=synthesize_fringe_scan(noisy_quantized_images)
    
    # 画像を再生成
    reconstructed_image_raw=reconstruct_image(g_FS,beta_prime,X,Y)

    # 結果の可視化
    visualize_coded_images(input_image_uint8, coded_images, phase_labels)
    visualize_synthesis_result(g_FS)
    visualize_final_result(input_image_uint8, reconstructed_image_raw)