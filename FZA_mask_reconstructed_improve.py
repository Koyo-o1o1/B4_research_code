import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

# 符号化画像の生成(式(2),(3))
def generate_code_images(R_ch,G_ch,B_ch,phases,beta_prime,X,Y):
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

        # 各カラーチャンネルで畳み込みを計算(式(3))
        coded_R=signal.fftconvolve(R_ch,h_FZA,mode='same')
        coded_G=signal.fftconvolve(G_ch,h_FZA,mode='same')
        coded_B=signal.fftconvolve(B_ch,h_FZA,mode='same')

        # 各カラーチャンネルを結合してカラー画像へ
        coded_color_image=np.stack([coded_R,coded_G,coded_B],axis=-1)
        coded_images.append(coded_color_image)
        i+=1

    print("畳み込み完了")
    return coded_images


# ホワイトガウスノイズを付加、12bit量子化
def add_noise_and_quantization(coded_images,snr_db):
    print(f"\n{snr_db}dBのノイズ付加および12bit量子化...")

    # 信号パワーの計算
    signal_power=np.mean(coded_images[0]**2)

    # ノイズパワーの計算とノイズ生成
    # snr(dB)からノイズの標準偏差を計算
    snr_linear=10**(snr_db/10)
    noise_power=signal_power/snr_linear
    noise_sigma=np.sqrt(noise_power)

    noisy_images=[]
    for img in coded_images:
        # 各画像にガウスノイズを付加
        noise=np.random.normal(0,noise_sigma,img.shape)
        noisy_images.append(img+noise)
    
    # 12bitへの量子化
    quantization_lebels=2**12-1

    # すべてのノイズ付加画像に共通のスケールを適用するため最大値最小値をもとめる
    global_min=min(np.min(img) for img in noisy_images)
    global_max=min(np.max(img) for img in noisy_images)

    quantized_images=[]
    for img in noisy_images:
        # データを0～1の範囲にスケーリング
        scaled_img=(img-global_min)/(global_max-global_min)
        # 0～2095(2**12-1)の範囲に変換し整数を丸める
        quantized_img_int=np.round(scaled_img*quantization_lebels)
        # 後の研鑽のためにfloat型へ
        quantized_images.append(quantized_img_int.astype(np.float64))
    
    print("処理終了")
    return quantized_images
    



# フリンジスキャン(式(6))
def synthesize_fringe_scan(coded_images):
    # ４枚の複素数画像を１枚に合成
    print("FS...")
    g_0,g_pi_half,g_pi,g_3pi_half=coded_images[0],coded_images[1],coded_images[2],coded_images[3]
    j=1j

    # 各カラーチャンネルにおいてフリンジスキャンを実行
    g_FS_R=-(g_pi_half[:,:,0]-g_3pi_half[:,:,0])+j*(g_0[:,:,0]-g_pi[:,:,0])
    g_FS_G=-(g_pi_half[:,:,1]-g_3pi_half[:,:,1])+j*(g_0[:,:,1]-g_pi[:,:,1])
    g_FS_B=-(g_pi_half[:,:,2]-g_3pi_half[:,:,2])+j*(g_0[:,:,2]-g_pi[:,:,2])

    print("FS完了")
    return g_FS_R,g_FS_G,g_FS_B


# 画像再構成(式(8),(9))
def reconstruct_image(g_FS_R,g_FS_G,g_FS_B,beta_prime,X,Y):
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
    G_FS_R=np.fft.fft2(g_FS_R)
    G_FS_G=np.fft.fft2(g_FS_G)
    G_FS_B=np.fft.fft2(g_FS_B)

    # 周波数領域で逆フィルタを適用
    F_reconstructed_R=G_FS_R*H_geo_inv
    F_reconstructed_G=G_FS_G*H_geo_inv
    F_reconstructed_B=G_FS_B*H_geo_inv

    # 逆フーリエ変換で空間領域へ戻す
    f_geo_R=np.fft.ifft2(F_reconstructed_R)
    f_geo_G=np.fft.ifft2(F_reconstructed_G)
    f_geo_B=np.fft.ifft2(F_reconstructed_B)

    # 絶対値を取り、3チャンネルを合成
    f_geo_R_abs=np.abs(f_geo_R)
    f_geo_G_abs=np.abs(f_geo_G)
    f_geo_B_abs=np.abs(f_geo_B)

    reconstructed_image_raw=np.stack([f_geo_R_abs,f_geo_G_abs,f_geo_B_abs],axis=-1)

    return reconstructed_image_raw


# -----------------------------------------------------------------------


# 可視化用関数

def visualize_psf(h_FZA,i):
    # PSFを画像として表示してエイリアシングを確認する
    plt.figure(figsize=(6, 6))
    plt.imshow(h_FZA, cmap='gray')
    if i==0:
        plt.title("PSF (h_FZA) for Aliasing Check (φ=0)")
        plt.colorbar()
        plt.savefig('img/FZA/FZA_phi_0.png')
        plt.show()
    elif i==1:
        plt.title("PSF (h_FZA) for Aliasing Check (φ=π/2)")
        plt.colorbar()
        plt.savefig('img/FZA/FZA_phi_pi_half.png')
        plt.show()
    elif i==2:
        plt.title("PSF (h_FZA) for Aliasing Check (φ=π)")
        plt.colorbar()
        plt.savefig('img/FZA/FZA_phi_pi.png')
        plt.show()
    else:
        plt.title("PSF (h_FZA) for Aliasing Check (φ=3π/2)")
        plt.colorbar()
        plt.savefig('img/FZA/FZA_phi_3pi_half.png')
        plt.show()


def visualize_coded_images(original_img,coded_images,phase_labels):
    # ４枚の符号化画像を可視化
    fig,axes=plt.subplots(1,5,figsize=(25,5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Color Image")
    for i, img in enumerate(coded_images):
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            img_normalized = (img - img_min) / (img_max - img_min)
            img_display = (img_normalized * 255).astype(np.uint8)
        else:
            img_display = img.astype(np.uint8)
        axes[i+1].imshow(img_display)
        axes[i+1].set_title(f"Coded Image ({phase_labels[i]})")
    plt.tight_layout()
    plt.savefig('img/coded_image/coded_images.png')
    plt.show()


def visualize_synthesis_result(g_FS_R, g_FS_G, g_FS_B):
    # 複素数画像の振幅と位相を可視化
    g_FS_R_magnitude = np.abs(g_FS_R)
    g_FS_G_magnitude = np.abs(g_FS_G)
    g_FS_B_magnitude = np.abs(g_FS_B)
    g_FS_R_phase = np.angle(g_FS_R)
    
    # --- 位相データを0～2πの範囲に変換 ---
    # np.angleは-π～+πを返すため、負の値に2πを足して0～2πに調整
    g_FS_R_phase[g_FS_R_phase < 0] += 2 * np.pi
    
    magnitude_color = np.zeros((g_FS_R.shape[0], g_FS_R.shape[1], 3), dtype=np.uint8)
    magnitude_color[:, :, 0] = 255 * (g_FS_R_magnitude / np.max(g_FS_R_magnitude))
    magnitude_color[:, :, 1] = 255 * (g_FS_G_magnitude / np.max(g_FS_G_magnitude))
    magnitude_color[:, :, 2] = 255 * (g_FS_B_magnitude / np.max(g_FS_B_magnitude))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("fringe_scan_after", fontsize=16)
    
    # --- カラーの振幅プロット ---
    im0 = axes[0].imshow(magnitude_color)
    axes[0].set_title("amplitude (color)")
    # こちらは正規化後の0-255画像なのでカラーバーはつけません

    # --- Rチャンネルの振幅プロットとカラーバー ---
    im1 = axes[1].imshow(g_FS_R_magnitude, cmap='gray')
    axes[1].set_title("amplitude(R)")
    fig.colorbar(im1, ax=axes[1], orientation='vertical').set_label('Magnitude (Raw Value)')

    # --- Rチャンネルの位相プロットとカラーバー ---
    im2 = axes[2].imshow(g_FS_R_phase, cmap='hsv', vmin=0, vmax=2*np.pi)
    axes[2].set_title("phase(R)")
    fig.colorbar(im2, ax=axes[2], orientation='vertical').set_label('Phase (radians)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('img/FS/fringe_scan_synthesis.png')
    plt.show()


def visualize_final_result(original_img, reconstructed_image_raw):
    # 最終的な再構成結果をオリジナル画像と比較して表示
    img_min, img_max = np.min(reconstructed_image_raw), np.max(reconstructed_image_raw)
    if img_max > img_min:
        img_normalized = (reconstructed_image_raw - img_min) / (img_max - img_min)
        reconstructed_display = (img_normalized * 255).astype(np.uint8)
    else:
        reconstructed_display = reconstructed_image_raw.astype(np.uint8)
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("compare", fontsize=16)
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[1].imshow(reconstructed_display)
    axes[1].set_title("Reconstructed Image (Conventional)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('img/final_image/final_reconstruction.png')
    plt.show()


# -----------------------------------------------------------------------



if __name__ == '__main__':
    # パラメータの設定
    image_size=512
    grid_range=3.0
    beta=14.5
    d=5.0
    z=280.0
    M=d/z
    beta_prime=beta/((M+1)**2)

    phases=[0,np.pi/2,np.pi,3*np.pi/2]
    phase_labels=["φ=0", "φ=π/2", "φ=π", "φ=3π/2"]

    x=np.linspace(-grid_range,grid_range,image_size)
    y=np.linspace(-grid_range,grid_range,image_size)
    X,Y=np.meshgrid(x,y)

    # 画像へのpath
    image_path="C:\\Users\\ko11s\\OneDrive\\Desktop\\research\\b\\img\\subject.jpg" 

    # カラー画像(512*512)を読み込み、float型に変換
    input_image_uint8=np.array(Image.open(image_path).convert("RGB"))
    input_image=input_image_uint8.astype(np.float64)

    # 入力画像をR, G, Bチャンネルに分離
    R_channel=input_image[:,:,0]
    G_channel=input_image[:,:,1]
    B_channel=input_image[:,:,2]

    # 処理を順に実行
    # 符号化画像を生成
    coded_images=generate_code_images(R_channel,G_channel,B_channel,phases,beta_prime,X,Y)
    
    # フリンジスキャンを実行
    g_FS_R,g_FS_G,g_FS_B=synthesize_fringe_scan(coded_images)
    
    # 画像を再生成
    reconstructed_image_raw=reconstruct_image(g_FS_R,g_FS_G,g_FS_B,beta_prime,X,Y)

    # 結果の可視化
    visualize_coded_images(input_image_uint8, coded_images, phase_labels)
    visualize_synthesis_result(g_FS_R, g_FS_G, g_FS_B)
    visualize_final_result(input_image_uint8, reconstructed_image_raw)