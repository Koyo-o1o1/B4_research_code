import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

# FZAのパラメータと描画範囲を設定
image_size=512 # 画像のサイズ
grid_range=3.0 # 空間座標の範囲
beta=25.0
d=5.0 # FZAからセンサまでの距離(mm)
z=280.0 # 被写体からFZAまでの距離
M=d/z
beta_prime=beta/((M+1)**2)

# 4つの位相{0,π/2,π,3π/2}
phases=[0,np.pi/2,np.pi,3*np.pi/2]
phase_labels=["φ=0","φ=π/2","φ=π","φ=3π/2"]
coded_images=[] # 生成した4枚の画像を格納するリスト

# 座標グリッドを作成
x=np.linspace(-grid_range,grid_range,image_size)
y=np.linspace(-grid_range,grid_range,image_size)

# 1次元配列から２次元の座標を作成
X,Y=np.meshgrid(x,y)

# 画像へのpath
image_path="C:\\Users\\ko11s\\OneDrive\\Desktop\\research\\b\\img\\subject.jpg" 

# カラー画像(512*512)を読み込み、float型に変換
input_image_uint8 = np.array(Image.open(image_path).convert("RGB"))
input_image = input_image_uint8.astype(np.float64)

# 入力画像をR, G, Bチャンネルに分離
R_channel = input_image[:, :, 0]
G_channel = input_image[:, :, 1]
B_channel = input_image[:, :, 2]

# 式(3)の実装

print("4つの位相で畳み込み処理...\n")
for phi in phases:
    # PSFをFZAの幾何学的影で近似(512*512) 式(2)
    h_FZA=0.5*(1+np.cos(beta_prime*(X**2+Y**2)-phi))

    # 各チャンネルで畳み込みを計算 式(3)
    coded_R=signal.fftconvolve(R_channel,h_FZA,mode='same')
    coded_G=signal.fftconvolve(G_channel,h_FZA,mode='same')
    coded_B=signal.fftconvolve(B_channel,h_FZA,mode='same')

    # 各チャンネルを結合してカラー画像に戻す
    coded_color_image=np.stack([coded_R,coded_G,coded_B],axis=-1)

    # 結果をリストに保存
    coded_images.append(coded_color_image)
print("処理完了")

# # --- 結果の可視化 ---
# fig, axes = plt.subplots(1, 5, figsize=(25, 5))

# # 元画像の表示
# axes[0].imshow(input_image_uint8)
# axes[0].set_title("Original Color Image")
# axes[0].grid(False)

# # 4枚の符号化画像の表示
# for i, img in enumerate(coded_images):
#     # 輝度値が大きな値になっているため、正規化して0-255の範囲に収める
#     img_min = np.min(img)
#     img_max = np.max(img)
#     # ゼロ除算を避けるためのチェック
#     if img_max > img_min:
#         img_normalized = (img - img_min) / (img_max - img_min)
#         img_display = (img_normalized * 255).astype(np.uint8)
#     else:
#         # 画像が真っ黒などの場合
#         img_display = img.astype(np.uint8)

#     axes[i+1].imshow(img_display)
#     axes[i+1].set_title(f"Coded Image ({phase_labels[i]})")
#     axes[i+1].grid(False)

# plt.tight_layout()
# plt.savefig('coded_images.png')
# plt.show()

# 4枚の符号化された生データを使って計算
g_0,g_pi_half,g_pi,g_3pi_half=coded_images[0],coded_images[1],coded_images[2],coded_images[3]
j=1j

# 式(6)
print("\nフリンジスキャンを実行...")
g_FS_R=-(g_pi_half[:, :, 0]-g_3pi_half[:, :, 0])+j*(g_0[:, :, 0]-g_pi[:, :, 0])
g_FS_G=-(g_pi_half[:, :, 1]-g_3pi_half[:, :, 1])+j*(g_0[:, :, 1]-g_pi[:, :, 1])
g_FS_B=-(g_pi_half[:, :, 2]-g_3pi_half[:, :, 2])+j*(g_0[:, :, 2]-g_pi[:, :, 2])
print("処理完了")

# # --- 結果を確認 ---
# print("\n--- 計算結果 ---")
# print(f"Rチャンネル合成後の形状: {g_FS_R.shape}, データ型: {g_FS_R.dtype}")

# # --- 結果を可視化 ---
# g_FS_R_magnitude = np.abs(g_FS_R)
# g_FS_G_magnitude = np.abs(g_FS_G)
# g_FS_B_magnitude = np.abs(g_FS_B)

# magnitude_color = np.zeros((image_size, image_size, 3), dtype=np.uint8)
# magnitude_color[:, :, 0] = 255 * (g_FS_R_magnitude / np.max(g_FS_R_magnitude))
# magnitude_color[:, :, 1] = 255 * (g_FS_G_magnitude / np.max(g_FS_G_magnitude))
# magnitude_color[:, :, 2] = 255 * (g_FS_B_magnitude / np.max(g_FS_B_magnitude))

# g_FS_R_phase = np.angle(g_FS_R)

# fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
# fig2.suptitle("fringescan_after", fontsize=16)
# axes2[0].imshow(magnitude_color)
# axes2[0].set_title("color")
# axes2[1].imshow(g_FS_R_magnitude, cmap='gray')
# axes2[1].set_title("amplitude (R)")
# axes2[2].imshow(g_FS_R_phase, cmap='hsv')
# axes2[2].set_title("phase (R)")
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('fringe_scan_synthesis.png')
# plt.show()


# 式(7),(8)の実装

# 単一複素数伝達関数の計算
print("\n単一複素数伝達関数を作成...")
h_geo = np.exp(j * beta_prime * (X**2 + Y**2)) # 式(7)

# h_geoをフーリエ変換してH_geoを得る
# fftshiftで画像中央を(0,0)としてフーリエ変換
H_geo = np.fft.fft2(np.fft.ifftshift(h_geo)) # 式(7)のフーリエ変換(式(8))

# 逆フィルタH_geo_invの作成(H_geoの複素共役)
H_geo_inv=np.conj(H_geo)

# 直接式(8)
# dx = (2 * grid_range) / image_size
# u = np.fft.fftfreq(image_size, d=dx)
# v = np.fft.fftfreq(image_size, d=dx)
# U, V = np.meshgrid(u, v)

# H_geo = np.exp(-j * (np.pi**2 / beta_prime) * (U**2 + V**2))
# H_geo_inv = np.conj(H_geo)
print("作成完了")

# 画像の再構成 式(9)

print("\n画像の再構成を実行...")
# 各カラーのフリンジスキャンのフーリエ変換(周波数領域へ)
G_FS_R = np.fft.fft2(g_FS_R)
G_FS_G = np.fft.fft2(g_FS_G)
G_FS_B = np.fft.fft2(g_FS_B)

# 周波数領域での積
F_reconstructed_R = H_geo_inv*G_FS_R
F_reconstructed_G = H_geo_inv*G_FS_G
F_reconstructed_B = H_geo_inv*G_FS_B

# 逆フーリエ
f_geo_R = np.fft.ifft2(F_reconstructed_R)
f_geo_G = np.fft.ifft2(F_reconstructed_G)
f_geo_B = np.fft.ifft2(F_reconstructed_B)
print("再構成完了")


# --- 最終画像の組み立てと可視化 ---
f_geo_R_abs = np.abs(f_geo_R)
f_geo_G_abs = np.abs(f_geo_G)
f_geo_B_abs = np.abs(f_geo_B)

reconstructed_image_raw = np.stack([f_geo_R_abs, f_geo_G_abs, f_geo_B_abs], axis=-1)


# 表示のために値を0-255の範囲に正規化
img_min = np.min(reconstructed_image_raw)
img_max = np.max(reconstructed_image_raw)
if img_max > img_min:
    img_normalized = (reconstructed_image_raw - img_min) / (img_max - img_min)
    reconstructed_image_display = (img_normalized * 255).astype(np.uint8)
else:
    reconstructed_image_display = reconstructed_image_raw.astype(np.uint8)

# --- 最終結果の表示 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("compare_Image", fontsize=16)

axes[0].imshow(input_image_uint8)
axes[0].set_title("Original Image")

axes[1].imshow(reconstructed_image_display)
axes[1].set_title("Reconstructed Image (Conventional)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('final_reconstruction.png')
plt.show()