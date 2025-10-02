# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from PIL import Image
# import os # フォルダ作成のために追加

# # リサイズ関数 (変更なし)
# def resize_to(img_2d, new_w, new_h, resample=Image.LANCZOS, dtype=np.float32):
#     pil = Image.fromarray(np.asarray(img_2d, dtype=np.float32), mode='F')
#     pil = pil.resize((new_w, new_h), resample=resample)
#     arr = np.asarray(pil, dtype=dtype)
#     return arr

# # 単一の符号化画像を生成する関数 (変更なし)
# def generate_single_coded_image(gray_image, phi, beta_prime, X, Y):
#     print("畳み込み処理 (単一画像)...")
#     h_FZA = 0.5 * (1 + np.cos(beta_prime * (X**2 + Y**2) - phi))
#     coded_image = signal.fftconvolve(gray_image, h_FZA, mode='same')
#     print("畳み込み完了")
#     return coded_image, h_FZA

# # ノイズ付加関数 (変更なし)
# def add_awgn(signal, snr_db):
#     signal_power = np.var(signal)
#     snr_linear = 10**(snr_db / 10)
#     noise_power = signal_power / snr_linear
#     noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
#     return signal + noise

# # 量子化関数 (変更なし)
# def quantize_to_12bit(signal):
#     sig_min, sig_max = np.min(signal), np.max(signal)
#     norm_signal = (signal - sig_min) / (sig_max - sig_min + 1e-12)
#     quantized = np.round(norm_signal * 4095)
#     dequantized = quantized / 4095.0
#     return dequantized * (sig_max - sig_min) + sig_min

# # 画像再構成関数 (変更なし)
# def reconstruct_image_no_fs(coded_image, psf, use_wiener=False, K=0.01):
#     print(f"デコンボリューション処理 (ウィーナーフィルタ: {'有効' if use_wiener else '無効'})...")
    
#     H = np.fft.fft2(psf)
#     G = np.fft.fft2(coded_image)
    
#     if use_wiener:
#         H_conj = np.conj(H)
#         H_abs_sq = np.abs(H)**2
#         H_wiener = H_conj / (H_abs_sq + K)
#         F_reconstructed = G * H_wiener
#     else:
#         epsilon = 1e-8
#         H_inv = np.where(np.abs(H) > epsilon, 1.0 / H, 0)
#         F_reconstructed = G * H_inv

#     f_reconstructed = np.fft.ifft2(F_reconstructed)
#     reconstructed_image = np.abs(f_reconstructed)
#     print("デコンボリューション完了")
    
#     return reconstructed_image

# # --- メイン処理 ---
# if __name__ == '__main__':
#     # --- 出力設定 ---
#     # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
#     USE_WIENER_FILTER = True # ウィーナーフィルタを有効にする
#     OUTPUT_FOLDER = "output_images" # 保存先フォルダ名
#     # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

#     # --- フォルダ作成 ---
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#     print(f"'{OUTPUT_FOLDER}' フォルダに画像を保存します。")

#     # SLMの仕様
#     slm_resolution_x, slm_resolution_y = 1024, 768
#     slm_pixel_pitch = 36e-6

#     slm_width = slm_resolution_x * slm_pixel_pitch
#     slm_height = slm_resolution_y * slm_pixel_pitch

#     x_slm = np.linspace(-slm_width / 2, slm_width / 2, slm_resolution_x)
#     y_slm = np.linspace(-slm_height / 2, slm_height / 2, slm_resolution_y)
#     X_slm, Y_slm = np.meshgrid(x_slm, y_slm)
    
#     # Sensorの仕様
#     sensor_resolution_x, sensor_resolution_y = 4104, 3006

#     # パラメータの設定
#     beta = 0.4e6
#     d = 0.5e-2
#     z = 2.5e-1
#     M = d / z
#     beta_prime = beta / ((M + 1)**2)
#     snr_db = 30.0
#     phi_0 = 0

#     # 被写体画像のパス
#     image_path = "C:\\Users\\ko11s\\OneDrive\\Desktop\\research\\b\\img\\subject.jpg"

#     input_image_uint8 = np.array(
#         Image.open(image_path)
#         .convert("L")
#         .resize((slm_resolution_x, slm_resolution_y))
#     )
#     input_image = input_image_uint8.astype(np.float64)

#     # 符号化画像を生成
#     coded_image_slm, psf_slm = generate_single_coded_image(
#         input_image, phi_0, beta_prime, X_slm, Y_slm
#     )

#     # センサ解像度へリサイズ
#     coded_image_sensor = resize_to(
#         coded_image_slm, sensor_resolution_x, sensor_resolution_y
#     )
#     psf_sensor = resize_to(
#         psf_slm, sensor_resolution_x, sensor_resolution_y
#     )
    
#     # AWGN & 量子化
#     img_noisy = add_awgn(coded_image_sensor, snr_db)
#     img_quantized = quantize_to_12bit(img_noisy)

#     # ウィーナーフィルタの定数KをSNRから計算
#     snr_linear = 10**(snr_db / 10.0)
#     K = 1 / snr_linear

#     # 画像を再構成
#     reconstructed_image = reconstruct_image_no_fs(
#         img_quantized, 
#         psf_sensor, 
#         use_wiener=USE_WIENER_FILTER, 
#         K=K
#     )

#     # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
#     # ★ 1. 最終再構成画像の保存 ★
#     # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
#     # 画像データを0-1の範囲に正規化
#     rec_min, rec_max = reconstructed_image.min(), reconstructed_image.max()
#     normalized_reconstruction = (reconstructed_image - rec_min) / (rec_max - rec_min)
    
#     # ファイルに保存
#     reconstruction_path = os.path.join(OUTPUT_FOLDER, "reconstructed_image.png")
#     plt.imsave(reconstruction_path, normalized_reconstruction, cmap='gray')
#     print(f"再構成画像を '{reconstruction_path}' に保存しました。")
    
#     # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
#     # ★ 2. 逆フィルタ H_notFS_inv の可視化と保存 ★
#     # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
#     # H_notFS^inv を計算（単純な逆フィルタ）
#     H = np.fft.fft2(psf_sensor)
#     epsilon = 1e-8
#     H_inv = np.where(np.abs(H) > epsilon, 1.0 / H, 0)

#     # 振幅（magnitude）を取得し、対数スケールに変換して見やすくする
#     magnitude_log = np.log1p(np.abs(H_inv))
    
#     # 周波数成分をシフトして、中心にDC成分（ゼロ周波数）が来るようにする
#     magnitude_log_shifted = np.fft.fftshift(magnitude_log)

#     # 0-1の範囲に正規化
#     mag_min, mag_max = magnitude_log_shifted.min(), magnitude_log_shifted.max()
#     normalized_magnitude = (magnitude_log_shifted - mag_min) / (mag_max - mag_min)

#     # ファイルに保存
#     filter_path = os.path.join(OUTPUT_FOLDER, "inverse_filter_magnitude.png")
#     plt.imsave(filter_path, normalized_magnitude, cmap='gray')
#     print(f"逆フィルタの振幅画像を '{filter_path}' に保存しました。")






import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import os # フォルダ作成のために追加


def create_psf(phi, beta_prime, X, Y):
    print(f"PSF (φ={phi}) を生成中...")
    h_FZA = 0.5 * (1 + np.cos(beta_prime * (X**2 + Y**2) - phi))
    return h_FZA

# OTF(H)を画像として保存する関数
def save_otf_as_image(H, filename="otf_magnitude.png"):
    """
    周波数領域のOTF(H)の振幅を、余白やラベルなしの画像ファイルとして保存する。
    """
    print(f"OTFの振幅画像を '{filename}' として保存中...")
    
    # fftshiftで低周波成分を中心に戻す
    H_shifted = np.fft.fftshift(H)
    
    # 振幅(絶対値)を計算
    amplitude = np.abs(H_shifted)
    
    # 対数スケールに変換して見やすくする
    log_amplitude = np.log1p(amplitude)
    
    # 0-1の範囲に正規化
    log_min, log_max = np.min(log_amplitude), np.max(log_amplitude)
    if log_max > log_min:
        normalized_log_amplitude = (log_amplitude - log_min) / (log_max - log_min)
    else:
        normalized_log_amplitude = np.zeros_like(log_amplitude)
        
    # 0-255の8bit整数に変換
    amplitude_uint8 = (normalized_log_amplitude * 255).astype(np.uint8)
    
    # Pillowを使って画像として保存
    pil_image = Image.fromarray(amplitude_uint8, mode='L')
    pil_image.save(filename)
    
    print("保存完了。")

# リサイズ関数 (変更なし)
def resize_to(img_2d, new_w, new_h, resample=Image.LANCZOS, dtype=np.float32):
    pil = Image.fromarray(np.asarray(img_2d, dtype=np.float32), mode='F')
    pil = pil.resize((new_w, new_h), resample=resample)
    arr = np.asarray(pil, dtype=dtype)
    return arr

# 符号化画像の生成関数 (変更なし)
def generate_code_images(gray_image,phases,beta_prime,X,Y):
    print("畳み込み処理...")
    coded_images=[]
    for phi in phases:
        h_FZA=0.5*(1+np.cos(beta_prime*(X**2+Y**2)-phi))
        coded_image=signal.fftconvolve(gray_image, h_FZA, mode='same')
        coded_images.append(coded_image)
    print("畳み込み完了")
    return coded_images

# ノイズ・量子化関連の関数 (変更なし)
def add_awgn(signal, snr_db):
    signal_power = np.var(signal)
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def quantize_to_12bit(signal):
    sig_min, sig_max = np.min(signal), np.max(signal)
    norm_signal = (signal - sig_min) / (sig_max - sig_min + 1e-12)
    quantized = np.round(norm_signal * 4095)
    dequantized = quantized / 4095.0
    return dequantized * (sig_max - sig_min) + sig_min

# フリンジスキャン関数 (変更なし)
def synthesize_fringe_scan(coded_images):
    print("FS...")
    g_0,g_pi_half,g_pi,g_3pi_half=coded_images[0],coded_images[1],coded_images[2],coded_images[3]
    j=1j
    g_FS=-(g_pi_half-g_3pi_half)+j*(g_0-g_pi)
    print("FS完了")
    return g_FS

# ★★★ 修正点 1: 関数名を変更し、リサイズではなく直接フィルタを生成 ★★★
def create_fs_filter(beta_prime, sensor_resolution_x, sensor_resolution_y, sensor_pixel_pitch):
    """
    センサーの解像度で直接、空間領域のFSフィルタ(h_geo)を生成する。
    """
    print("FSフィルタをセンサー解像度で直接生成...")
    sensor_width = sensor_resolution_x * sensor_pixel_pitch
    sensor_height = sensor_resolution_y * sensor_pixel_pitch
    
    x_sensor = np.linspace(-sensor_width / 2, sensor_width / 2, sensor_resolution_x)
    y_sensor = np.linspace(-sensor_height / 2, sensor_height / 2, sensor_resolution_y)
    X_sensor, Y_sensor = np.meshgrid(x_sensor, y_sensor)
    
    h_geo_sensor = np.exp(1j * beta_prime * (X_sensor**2 + Y_sensor**2)).astype(np.complex64)
    
    return h_geo_sensor

# ★★★ 修正点 2: 逆フィルタの計算を明確化 ★★★
def reconstruct_image(g_FS_sensor, h_geo_sensor):
    """
    h_geo_sensor (空間フィルタ) を使って画像を再構成する
    """
    # 空間フィルタ h_geo から周波数フィルタ H_geo (OTF) を計算
    H_geo_OTF = np.fft.fft2(np.fft.ifftshift(h_geo_sensor))
    
    # 逆フィルタを作成 (論文 Eq. (9) より、逆フィルタは複素共役)
    H_geo_inv = np.conj(H_geo_OTF)

    # 画像再構成
    G_FS_sensor = np.fft.fft2(g_FS_sensor)
    F_reconstructed_sensor = G_FS_sensor * H_geo_inv
    f_geo_sensor = np.fft.ifft2(F_reconstructed_sensor)
    reconstructed_image_raw = np.abs(f_geo_sensor)
    
    return reconstructed_image_raw, H_geo_inv

# ★★★ 新しく追加したフィルタ可視化・保存関数 ★★★
def visualize_and_save_filter(inv_filter, folder):
    print("逆フィルタの可視化画像を生成・保存します...")
    # ゼロ周波数を中心にシフト
    filter_shifted = np.fft.fftshift(inv_filter)
    
    # 1. 振幅の可視化と保存
    magnitude = np.abs(filter_shifted)
    # 0-1に正規化
    mag_min, mag_max = magnitude.min(), magnitude.max()
    magnitude_normalized = (magnitude - mag_min) / (mag_max - mag_min)
    mag_path = os.path.join(folder, "FS_filter_magnitude.png")
    plt.imsave(mag_path, magnitude_normalized, cmap='gray')
    print(f"フィルタの振幅画像を '{mag_path}' に保存しました。")

    # 2. 位相の可視化と保存
    phase = np.angle(filter_shifted)
    # 0-1に正規化
    phase_min, phase_max = phase.min(), phase.max()
    phase_normalized = (phase - phase_min) / (phase_max - phase_min)
    phase_path = os.path.join(folder, "FS_filter_phase.png")
    plt.imsave(phase_path, phase_normalized, cmap='gray')
    print(f"フィルタの位相画像を '{phase_path}' に保存しました。")
    
# 可視化用関数は、簡単のためメイン処理から削除します

# --- メイン処理 ---
if __name__ == '__main__':
    OUTPUT_FOLDER = "output_images"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # SLMの仕様
    slm_resolution_x,slm_resolution_y=1024,768
    slm_pixel_pitch=36e-6
    slm_width=slm_resolution_x*slm_pixel_pitch
    slm_height=slm_resolution_y*slm_pixel_pitch
    x_slm=np.linspace(-slm_width/2,slm_width/2,slm_resolution_x)
    y_slm=np.linspace(-slm_height/2,slm_height/2,slm_resolution_y)
    X_slm,Y_slm=np.meshgrid(x_slm,y_slm)
    
    # Sensorの仕様
    sensor_resolution_x,sensor_resolution_y=4104,3006

    # パラメータの設定
    beta=0.4e6
    d=0.5e-2
    z=2.5e-1
    M=d/z
    beta_prime=beta/((M+1)**2)
    snr_db=30.0
    phases=[0,np.pi/2,np.pi,3*np.pi/2]

    # 被写体画像のパス
    image_path="C:\\Users\\ko11s\\OneDrive\\Desktop\\research\\b\\img\\subject.jpg" 

    input_image_uint8 = np.array(
        Image.open(image_path)
        .convert("L")
        .resize((slm_resolution_x, slm_resolution_y))
    )
    input_image = input_image_uint8.astype(np.float64)

    # 処理を順に実行
    coded_images_slm=generate_code_images(input_image,phases,beta_prime,X_slm,Y_slm)

    coded_images_sensor = [
        resize_to(img, sensor_resolution_x, sensor_resolution_y)
        for img in coded_images_slm
    ]

    noisy_quantized_images_sensor = []
    for img in coded_images_sensor:
        img_noisy = add_awgn(img, snr_db)
        img_quantized = quantize_to_12bit(img_noisy)
        noisy_quantized_images_sensor.append(img_quantized)

    g_FS_sensor=synthesize_fringe_scan(noisy_quantized_images_sensor)
    sensor_pixel_pitch=3.45e-6
    h_geo_sensor = create_fs_filter(beta_prime, sensor_resolution_x, sensor_resolution_y, sensor_pixel_pitch)
    
    # ★★★ 再構成関数から逆フィルタも受け取る ★★★
    reconstructed_image_raw, H_inv_filter = reconstruct_image(g_FS_sensor,h_geo_sensor)

    # ★★★ フィルタの可視化・保存関数を呼び出す ★★★
    # visualize_and_save_filter(H_inv_filter, OUTPUT_FOLDER)

    # ★★★ 最終再構成画像の保存 ★★★
    rec_min, rec_max = reconstructed_image_raw.min(), reconstructed_image_raw.max()
    normalized_reconstruction = (reconstructed_image_raw - rec_min) / (rec_max - rec_min)
    reconstruction_path = os.path.join(OUTPUT_FOLDER, "FS_reconstructed_image.png")
    # plt.imsave(reconstruction_path, normalized_reconstruction, cmap='gray')
    print(f"最終再構成画像を '{reconstruction_path}' に保存しました。")


    # --- 計算と保存 ---
    # 1. 空間領域のPSF(h_FZA)を生成
    phi_0 = 0
    psf = create_psf(phi_0, beta_prime, X_slm, Y_slm)

    # 2. PSFをフーリエ変換してOTF(H_notFS)を生成
    #    (論文のH_notFSに相当)
    OTF = np.fft.fft2(psf)

    # 3. OTFの振幅を画像として保存
    otf_path = os.path.join(OUTPUT_FOLDER, "otf_notFS_magnitude.png")
    save_otf_as_image(OTF, filename=otf_path)