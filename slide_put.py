import cv2
import numpy as np
import os

def save_fourier_spectrum(image_path, output_dir="output_images"):
    """
    画像をフーリエ変換し、パワースペクトル画像のみをファイルに保存する。

    Args:
        image_path (str): 処理対象の画像ファイルへのパス。
        output_dir (str): 出力画像を保存するディレクトリ名。
    """
    # --- 1. 出力ディレクトリを作成 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 2. 画像をグレースケールで読み込む ---
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"エラー: 画像 '{image_path}' を読み込めませんでした。")
        return

    # --- 3. 2次元フーリエ変換とパワースペクトルの計算 ---
    # 変換、シフト、対数スケールへの変換を一度に行う
    f_transform = np.fft.fft2(img)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)

    # --- 4. 画像として保存するために0-255の範囲に正規化 ---
    # cv2.imwriteで保存するために、浮動小数点数の配列を
    # 8ビットのグレースケール画像（0から255の整数値）に変換します。
    cv2.normalize(magnitude_spectrum, magnitude_spectrum, 0, 255, cv2.NORM_MINMAX)
    spectrum_image = np.uint8(magnitude_spectrum)

    # --- 5. 画像をファイルとして保存 ---
    # 保存するファイル名を生成
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(output_dir, f"{base_name}_spectrum.png")

    # cv2.imwrite()でNumPy配列を直接画像ファイルとして保存
    cv2.imwrite(output_filename, spectrum_image)
    print(f"パワースペクトル画像を '{output_filename}' に保存しました。")

# --- プログラムの実行 ---
if __name__ == '__main__':
    input_image_path = 'coded_image_0.jpg'
    save_fourier_spectrum(input_image_path)