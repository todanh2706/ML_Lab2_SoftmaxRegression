import os
import numpy as np
import matplotlib.pyplot as plt

from features import FeatureExtractor
from main import load_mnist_data  # hoặc từ file chứa hàm này, chỉnh lại import nếu cần


def visualize_block_averaging_example(
    idx: int = 0,
    block_size=(4, 4),
    save_path: str = "block_feature_demo.png"
):
    """
    Vẽ minh hoạ cho đặc trưng Block Averaging:
      (a) Ảnh gốc 28x28
      (b) Ảnh với lưới block 7x7 (mỗi block 4x4)
      (c) Ảnh sau khi đã average block (phiên bản 7x7 upsample lại 28x28)
      (d) Vector đặc trưng chiều 49 sau khi flatten

    Args:
        idx       : chỉ số ảnh trong tập train để minh hoạ
        block_size: kích thước block (bh, bw), mặc định (4,4) cho MNIST 28x28
        save_path : đường dẫn file ảnh để lưu minh hoạ
    """
    # 1. Load dữ liệu MNIST
    X_train_raw, y_train, X_test_raw, y_test = load_mnist_data()
    if X_train_raw is None:
        raise RuntimeError("Không load được dữ liệu MNIST, kiểm tra lại đường dẫn và cấu hình.")

    # Lấy 1 ảnh để minh hoạ
    img = X_train_raw[idx]  # shape (28, 28), kiểu uint8
    label = y_train[idx]

    bh, bw = block_size
    H, W = img.shape
    assert H % bh == 0 and W % bw == 0, "Kích thước ảnh phải chia hết cho kích thước block."
    new_h, new_w = H // bh, W // bw  # ví dụ 7x7

    # 2. Tính đặc trưng Block Averaging bằng hàm có sẵn
    block_feats = FeatureExtractor.get_block_features(
        img[np.newaxis, ...],  # shape (1, 28, 28)
        block_size=block_size
    )  # shape (1, new_h*new_w)
    block_vec = block_feats[0]          # shape (49,)
    block_map = block_vec.reshape(new_h, new_w)  # shape (7, 7)

    # 3. Upsample ảnh 7x7 block_map lên 28x28 để trực quan hoá
    #    Mỗi giá trị block được lặp lại thành patch 4x4
    block_upsampled = np.kron(block_map, np.ones((bh, bw)))

    # 4. Vẽ các hình
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    # (a) Ảnh gốc
    ax = axes[0]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"(a) Ảnh gốc\nlabel = {label}")
    ax.axis("off")

    # (b) Ảnh gốc với lưới block
    ax = axes[1]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"(b) Ảnh với lưới\n{new_h}x{new_w} block ({bh}x{bw})")
    ax.axis("off")

    # Vẽ lưới: đường kẻ chia block
    for r in range(1, new_h):
        y_line = r * bh - 0.5
        ax.axhline(y_line, linestyle="--", linewidth=0.7)
    for c in range(1, new_w):
        x_line = c * bw - 0.5
        ax.axvline(x_line, linestyle="--", linewidth=0.7)

    # (c) Ảnh sau khi average block (phiên bản 7x7 được phóng to)
    ax = axes[2]
    ax.imshow(block_upsampled, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("(c) Ảnh sau Block Averaging\n(7x7 upsample lên 28x28)")
    ax.axis("off")

    # (d) Vector đặc trưng chiều 49
    ax = axes[3]
    ax.bar(np.arange(block_vec.size), block_vec)
    ax.set_title("(d) Vector đặc trưng (49 chiều)")
    ax.set_xlabel("Chỉ số phần tử")
    ax.set_ylabel("Giá trị trung bình block")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    plt.tight_layout()

    # 5. Lưu hình để chèn vào LaTeX
    plt.savefig(save_path, dpi=300)
    print(f"Đã lưu hình minh hoạ Block Averaging tại: {os.path.abspath(save_path)}")

    # Nếu muốn show luôn:
    # plt.show()


if __name__ == "__main__":
    # Ví dụ: minh hoạ với ảnh đầu tiên trong tập train
    visualize_block_averaging_example(idx=0, block_size=(4, 4))
