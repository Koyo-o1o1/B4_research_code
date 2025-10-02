from PIL import Image, ImageDraw

# 画像のサイズを指定
width = 1024
height = 768

# 点光源の半径を指定
radius = 5  # 円の半径を小さくして点に見せる

# 黒い背景の画像を新規作成
# 'RGB'はカラーモード、(width, height)はサイズ、'black'は背景色
image = Image.new('RGB', (width, height), 'black')

# 描画オブジェクトを作成
draw = ImageDraw.Draw(image)

# 画像の中心座標を計算
center_x = width // 2
center_y = height // 2

# 円を描画するための bounding box (左上と右下の座標) を計算
left = center_x - radius
top = center_y - radius
right = center_x + radius
bottom = center_y + radius
bounding_box = [left, top, right, bottom]

# 中心の円（点光源）を白で描画
draw.ellipse(bounding_box, fill='white')

# 画像をファイルとして保存
image.save('point_light_source.png')

print("画像 'point_light_source.png' を作成しました。")