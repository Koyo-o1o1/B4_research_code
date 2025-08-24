import numpy as np
import matplotlib.pyplot as plt

# FZAのパラメータと描画範囲を設定
# 画像のサイズ
image_size=512

# 空間座標の範囲
grid_range=3.0

# FZAのパラメータ
beta=25.0
phi=0*np.pi # FZAの初期位相
d=5.0 # FZAからセンサまでの距離(mm)
z=280.0 # 被写体からFZAまでの距離

# 座標グリッドを作成
x=np.linspace(-grid_range,grid_range,image_size)
y=np.linspace(-grid_range,grid_range,image_size)

# 1次元配列から２次元の座標を作成
Xp,Yp=np.meshgrid(x,y)

# FZAの透過率を計算(512*512)
transmittance=0.5*(1+np.cos(beta*(Xp**2+Yp**2)-phi))

# 計算結果を画像として表示
plt.figure(figsize=(8,6))
# 2次元配列を画像として表示(data,color,extent=[x_min,x_max,y_min,y_max])
plt.imshow(transmittance, cmap='gray', extent=[-grid_range, grid_range, -grid_range, grid_range])

# カラーバー、タイトル、軸ラベルを付与
plt.colorbar(label='Transmittance')
plt.title(f'Fresnel Zone Aperture (FZA) Pattern\n(β={beta},φ={phi})')
plt.xlabel('Xp coordinate')
plt.ylabel('Yp coordinate')
# グリッド線は表示しない
plt.grid(False)

# 画像をファイルに保存
plt.savefig('fza_pattern_0.png')
# 画面にプロットを表示
plt.show()