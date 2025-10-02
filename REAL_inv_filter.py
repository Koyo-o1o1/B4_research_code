import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# def check_psf_histogram(image_path):
#     """
#     指定されたPSF画像のヒストグラムを可視化して、輝度値の分布を確認する。
#     """
#     try:
#         image = Image.open(image_path).convert("L")
#         image_data = np.array(image)
#         print(f"'{image_path}' を読み込みました。")
#     except FileNotFoundError:
#         print(f"エラー: ファイル '{image_path}' が見つかりません。")
#         return

#     plt.figure(figsize=(10, 6))
#     plt.hist(image_data.ravel(), bins=256, range=(0, 255), color='gray')
#     plt.title(f"Histogram of {os.path.basename(image_path)}")
#     plt.xlabel("Pixel Brightness (0=Black, 255=White)")
#     plt.ylabel("Number of Pixels")
#     plt.grid(True)
#     plt.xlim(0, 255)
#     plt.show()

# FS
def synthesize_fringe_scan(coded_images):
    print("FS合成中...")
    print(len(coded_images))
    g_0, g_90, g_180, g_270 = coded_images
    j = 1j
    g_FS = -(g_90 - g_270) + j * (g_0 - g_180)
    print("FS合成完了")
    return g_FS

# 光学伝達関数を計算
def create_otf_from_measured_psf(psf_measured):
    print("計測されたPSFからOTF(H_geo)を生成中...")
    H_geo = np.fft.fft2(np.fft.ifftshift(psf_measured))
    print("OTF(H_geo)生成完了")
    return H_geo

# def visualize_otf_amplitude_profile(H_geo):
#     """
#     OTF(H_geo)の振幅スペクトルのラインプロファイルを作成・表示する。
#     ゼロ周波数を正しく1に規格化するため、奇数サイズにクロップして処理。
#     """
#     print("振幅スペクトルのラインプロファイルを生成中...")

#     # 偶数サイズなら奇数にクロップ
#     h, w = H_geo.shape
#     if h % 2 == 0:
#         H_geo = H_geo[1:, :]
#         h -= 1
#     if w % 2 == 0:
#         H_geo = H_geo[:, 1:]
#         w -= 1

#     # FFTシフトして中心を取得
#     H_shifted = np.fft.fftshift(H_geo)
#     amplitude = np.abs(H_shifted)

#     center_y, center_x = h // 2, w // 2
#     zero_freq_val = amplitude[center_y, center_x]

#     if zero_freq_val > 0:
#         normalized_amplitude = amplitude / zero_freq_val
#     else:
#         normalized_amplitude = amplitude

#     # 中心から右方向のラインプロファイル
#     line_profile = normalized_amplitude[center_y, center_x:]

#     plt.figure(figsize=(12, 7))
#     plt.plot(line_profile)
#     plt.title("OTF Amplitude Spectrum Profile (Center to Edge, Odd-sized)")
#     plt.xlabel("Spatial Frequency (pixels from center)")
#     plt.ylabel("Normalized Amplitude (MTF)")
#     plt.grid(True)
#     plt.ylim(0, 1.1)

#     profile_filename = "otf_amplitude_profile_fixed.png"
#     plt.savefig(profile_filename)
#     print(f" -> '{profile_filename}' にプロファイルを保存しました。")
#     plt.show()


def visualize_otf_amplitude_profile_limited(H_geo, limit):
    print(f"振幅スペクトルのラインプロファイルを生成中 (0-{limit-1}の範囲)...")

    # 偶数サイズなら奇数にクロップ
    h, w = H_geo.shape
    if h % 2 == 0:
        H_geo = H_geo[1:, :]
        h -= 1
    if w % 2 == 0:
        H_geo = H_geo[:, 1:]
        w -= 1

    # FFTシフトして中心を取得
    H_shifted = np.fft.fftshift(H_geo)
    amplitude = np.abs(H_shifted)

    center_y, center_x = h // 2, w // 2
    zero_freq_val = amplitude[center_y, center_x]

    if zero_freq_val > 0:
        normalized_amplitude = amplitude / zero_freq_val
    else:
        normalized_amplitude = amplitude

    # 中心から右方向のラインプロファイル
    line_profile = normalized_amplitude[center_y, center_x:]
    
    # プロット範囲を制限
    line_profile_limited = line_profile[:limit]

    plt.figure(figsize=(12, 7))
    plt.plot(line_profile_limited)
    plt.title(f"OTF Amplitude Spectrum Profile (First {limit} Frequencies)")
    plt.xlabel("Spatial Frequency (pixels from center)")
    plt.ylabel("Normalized Amplitude (MTF)")
    plt.grid(True)
    plt.ylim(0, 1.1)
    # ★X軸の表示範囲もデータに合わせて設定
    plt.xlim(0, limit - 1)

    profile_filename = "otf_amplitude_profile_limited.png"
    # plt.savefig(profile_filename)
    print(f" -> '{profile_filename}' にプロファイルを保存しました。")
    plt.show()



def visualize_fringe_scan_result(complex_image, save_prefix="fringe_scan_result"):
    """
    フリンジスキャン後の複素画像の振幅と位相を可視化して保存する。
    """
    print("フリンジスキャン結果を可視化中...")
    
    amplitude = np.abs(complex_image)
    phase = np.angle(complex_image)
    phase[phase < 0] += 2 * np.pi

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Fringe Scan Synthesis Result")

    # 振幅のプロット
    im1 = axes[0].imshow(amplitude, cmap='gray')
    axes[0].set_title("Amplitude")
    fig.colorbar(im1, ax=axes[0])

    # 位相のプロット
    im2 = axes[1].imshow(phase, cmap='hsv', vmin=0, vmax=2 * np.pi)
    axes[1].set_title("Phase")
    cbar = fig.colorbar(im2, ax=axes[1])
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(f"{save_prefix}.png")
    plt.show()

# 再構成を行う関数を追加
def reconstruct_image(captured_g_fs, measured_otf):
    print("画像再構成中...")
    
    # 計測されたOTFから逆フィルタを作成 (式(9)より H_inv = H*)
    H_inv = np.conj(measured_otf)

    # 撮影された画像をフーリエ変換
    G_FS = np.fft.fft2(np.fft.ifftshift(captured_g_fs))

    # 周波数領域で逆フィルタを適用
    F_reconstructed = G_FS * H_inv

    # 逆フーリエ変換で空間領域へ戻す
    f_reconstructed_shifted = np.fft.ifft2(F_reconstructed)
    
    # 像が中心に来るようにシフト
    f_reconstructed = np.fft.fftshift(f_reconstructed_shifted)

    # 絶対値を取って再構成画像とする
    reconstructed_image = np.abs(f_reconstructed)
    
    print("画像再構成完了")
    return reconstructed_image

# 再構成画像を可視化する関数
def visualize_reconstruction(image, save_name="reconstructed_point_source.png"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title("Reconstructed Image")
    plt.colorbar()
    # plt.savefig(save_name)
    print(f" -> '{save_name}' に再構成画像を保存しました。")
    plt.show()

if __name__ == '__main__':
    
    # 撮影したPSF画像が保存されているフォルダ
    psf_folder = "IDS_img"
    psf_filenames = [os.path.join(psf_folder, f"PSF_{phi}.png") for phi in [0, 90, 180, 270]]
    
    psf_images = [np.array(Image.open(f).convert("L"), dtype=np.float64) for f in psf_filenames]

    # FS
    g_FS = synthesize_fringe_scan(psf_images)

    visualize_fringe_scan_result(g_FS, save_prefix="measured_psf_visualization")

    # 計測されたPSFからOTF(H_geo)を生成
    H_geo_final = create_otf_from_measured_psf(g_FS)

    # OTFをファイルに保存
    otf_filename = "measured_otf.npy"
    # np.save(otf_filename, H_geo_final)
    print(f"\nOTF(H_geo)を '{otf_filename}' として保存しました。")

    # OTFの振幅スペクトルのラインプロファイルを可視化(80まで可視化)
    # visualize_otf_amplitude_profile(H_geo_final)
    visualize_otf_amplitude_profile_limited(H_geo_final, limit=80)

    # g_FSをH_geo_finalで再構成
    reconstructed_img = reconstruct_image(g_FS, H_geo_final)
    
    # 再構成結果を可視化
    visualize_reconstruction(reconstructed_img)