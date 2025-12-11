import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.restoration import denoise_tv_chambolle # L1ベースの全変動正則化
from scipy.ndimage import gaussian_filter # L2ベースの平滑化 (ぼかし)
from IPython.display import clear_output
import time

# ----------------------------------------------------------------------
# 1. 画像の準備
# ----------------------------------------------------------------------
# 元の画像を取得 (scikit-imageのサンプル画像を使用)
# URLから直接画像を読み込むことも可能
# 例: image = io.imread('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/800px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
try:
    image = io.imread('https://raw.githubusercontent.com/scikit-image/scikit-image/main/skimage/data/chelsea.png')
except:
    print("サンプル画像のダウンロードに失敗しました。代替画像を使用します。")
    # 代替として乱数で生成したドットパターン画像を使用
    image = np.zeros((128, 128))
    for _ in range(20):
        x, y = np.random.randint(0, 128, 2)
        image[y:y+np.random.randint(1,5), x:x+np.random.randint(1,5)] = np.random.rand()
    image = (image * 255).astype(np.uint8)


# 画像を浮動小数点数に変換 (0-1の範囲)
image_float = img_as_float(image)

# 画像にノイズを追加 (ガウシアンノイズ)
# ノイズがL1/L2でどう処理されるかを見るため
noise_sigma = 0.15 # ノイズの強度
noisy_image = image_float + noise_sigma * np.random.randn(*image_float.shape)
noisy_image = np.clip(noisy_image, 0, 1) # 値を0-1にクリップ

# ----------------------------------------------------------------------
# 2. L1ノルムとL2ノルムに基づく画像処理
# ----------------------------------------------------------------------

# (1) OLS (原画像またはノイズ画像)
# OLSは画像処理では「そのまま」という意味合いが強い
original_display = image_float
noisy_display = noisy_image

# (2) L2ノルムに基づく処理 (ぼかし/平滑化)
# ガウシアンフィルタが代表的。全体を滑らかにする。
# L2正則化は画像の勾配のL2ノルムを最小化するTikhonov正則化などもあるが、
# 視覚的な分かりやすさからガウシアンフィルタを用いる
sigma_l2 = 2.0 # 平滑化の強さ (ぼかし具合)
blurred_l2_image = gaussian_filter(noisy_image, sigma=sigma_l2)

# (3) L1ノルムに基づく処理 (全変動正則化 - エッジ保存ノイズ除去)
# L1ノルムが勾配（差分）に適用され、エッジを保ちつつノイズを除去する
weight_l1 = 0.2 # ノイズ除去の強さ (L1ペナルティの重み)
# 'multichannel'引数を削除し、画像がカラーであってもモノクロとして処理（TV正則化の一般的な手法）
denoised_l1_image = denoise_tv_chambolle(noisy_image, weight=weight_l1)


# ----------------------------------------------------------------------
# 3. 結果の可視化 (画像出力)
# ----------------------------------------------------------------------
# fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # <--- この行をコメントアウト
# ax = axes.ravel()                               # <--- この行をコメントアウト

# # 元の画像
# ax[0].imshow(original_display, cmap='gray' if image.ndim == 2 else None)
# ax[0].set_title('Original Image (OLS Baseline)')
# ax[0].axis('off')

# # ノイズが乗った画像 (比較用)
# ax[1].imshow(noisy_display, cmap='gray' if image.ndim == 2 else None)
# ax[1].set_title('Noisy Image')
# ax[1].axis('off')

# # L2ノルムに基づく処理 (ガウシアンぼかし)
# ax[2].imshow(blurred_l2_image, cmap='gray' if image.ndim == 2 else None)
# ax[2].set_title(f'L2-like (Gaussian Blur, $\sigma$={sigma_l2})')
# ax[2].axis('off')

# # L1ノルムに基づく処理 (全変動正則化)
# ax[3].imshow(denoised_l1_image, cmap='gray' if image.ndim == 2 else None)
# ax[3].set_title(f'L1-like (Total Variation, weight={weight_l1})')
# ax[3].axis('off')

# plt.tight_layout()
# plt.show() # <--- この行をコメントアウト

# ----------------------------------------------------------------------
# 4. (オプション) ドット画像でのL1/L2比較
# ----------------------------------------------------------------------
print("\n--- ドット画像でのL1/L2比較 ---")
dot_image_size = 64
dot_image = np.zeros((dot_image_size, dot_image_size))
# いくつかのドットを配置
for _ in range(5):
    x, y = np.random.randint(0, dot_image_size, 2)
    dot_image[y:y+2, x:x+2] = 1.0 # 濃いドット

# ドット画像にノイズを追加
noisy_dot_image = dot_image + 0.3 * np.random.randn(*dot_image.shape)
noisy_dot_image = np.clip(noisy_dot_image, 0, 1)

# L2処理
blurred_dot_l2 = gaussian_filter(noisy_dot_image, sigma=1.0) # 比較的強めのぼかし

# L1処理
# 'multichannel=False' を削除
denoised_dot_l1 = denoise_tv_chambolle(noisy_dot_image, weight=0.1)

fig_dots, axes_dots = plt.subplots(1, 4, figsize=(16, 4))
ax_dots = axes_dots.ravel()

ax_dots[0].imshow(dot_image, cmap='gray')
ax_dots[0].set_title('Original Dots')
ax_dots[0].axis('off')

ax_dots[1].imshow(noisy_dot_image, cmap='gray')
ax_dots[1].set_title('Noisy Dots')
ax_dots[1].axis('off')

ax_dots[2].imshow(blurred_dot_l2, cmap='gray')
ax_dots[2].set_title('Dots L2-like (Gaussian Blur)')
ax_dots[2].axis('off')

ax_dots[3].imshow(denoised_dot_l1, cmap='gray')
ax_dots[3].set_title('Dots L1-like (Total Variation)')
ax_dots[3].axis('off')

plt.tight_layout()
plt.show()
