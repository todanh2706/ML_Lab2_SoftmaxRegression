import numpy as np
import os
import time
from model import SoftmaxRegression
from features import FeatureExtractor
from data import train_val_split, load_mnist_npz
from dataLoader import MnistDataloader

# Cấu hình Hyperparameters
CONFIG = {
    'lr': 0.1,
    'reg': 1e-4,
    'epochs': 20,
    'batch_size': 128,
    'n_classes': 10,
    'val_split': 0.1  # 10% tập train dùng để validate
}

def load_mnist_data():
    """Load MNIST via tensorflow if available; fallback to notebook IDX loader or mnist.npz."""
    try:
        from tensorflow.keras.datasets import mnist
        (X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()
        return X_train_raw, y_train, X_test_raw, y_test
    except ImportError:
        print("TensorFlow not found; trying local IDX files via MnistDataloader.")
    except Exception as e:
        print(f"TensorFlow MNIST load failed: {e}; trying local IDX files.")

    def _candidate_sets():
        base_dirs = [
            os.getenv("MNIST_IDX_DIR"),
            os.path.join(os.getcwd(), "data"),
            os.path.join(os.getcwd(), "input"),
            os.path.join(os.getcwd(), "../input"),
            os.getcwd(),
        ]
        # Prefer actual .idx* files, then folder-wrapped versions.
        filename_sets = [
            (
                "train-images.idx3-ubyte",
                "train-labels.idx1-ubyte",
                "t10k-images.idx3-ubyte",
                "t10k-labels.idx1-ubyte",
            ),
            (
                "train-images-idx3-ubyte",
                "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte",
            ),
            (
                "train-images-idx3-ubyte/train-images-idx3-ubyte",
                "train-labels-idx1-ubyte/train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
            ),
        ]
        for base in base_dirs:
            if not base:
                continue
            for names in filename_sets:
                yield {
                    "train_images": os.path.join(base, names[0]),
                    "train_labels": os.path.join(base, names[1]),
                    "test_images": os.path.join(base, names[2]),
                    "test_labels": os.path.join(base, names[3]),
                }

    chosen = None
    for candidate in _candidate_sets():
        if all(os.path.isfile(p) for p in candidate.values()):
            chosen = candidate
            break

    if chosen is None:
        print("Error: IDX files not found (set MNIST_IDX_DIR or place files under data/).")
        local_npz = os.path.join(os.getcwd(), "mnist.npz")
        if os.path.exists(local_npz):
            try:
                return load_mnist_npz(local_npz)
            except Exception as e:
                print(f"Error loading mnist.npz fallback: {e}")
        return None, None, None, None

    mnist_dataloader = MnistDataloader(
        chosen["train_images"],
        chosen["train_labels"],
        chosen["test_images"],
        chosen["test_labels"],
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    return (
        np.array(x_train, dtype=np.uint8),
        np.array(y_train, dtype=np.uint8),
        np.array(x_test, dtype=np.uint8),
        np.array(y_test, dtype=np.uint8),
    )

def train_and_evaluate(feature_name, extract_func, X_raw, y, X_test_raw, y_test):
    print(f"\n{'='*10} ĐANG XỬ LÝ: {feature_name} {'='*10}")
    
    # 1. Trích xuất đặc trưng
    print(f"[*] Đang trích xuất đặc trưng (Feature Extraction)...")
    start_time = time.time()
    X_feat = extract_func(X_raw)
    X_test_feat = extract_func(X_test_raw)
    print(f"    - Thời gian trích xuất: {time.time() - start_time:.2f}s")
    print(f"    - Kích thước vector đặc trưng: {X_feat.shape[1]}")

    # 2. Chia tập train val 
    X_train_split, y_train_split, X_val_split, y_val_split = train_val_split(
        X_feat, y, val_ratio=CONFIG['val_split'], shuffle=True, seed=42
    )

    # 3. Khởi tạo Model
    model = SoftmaxRegression(
        n_features=X_feat.shape[1],
        n_classes=CONFIG['n_classes'],
        lr=CONFIG['lr'],
        reg=CONFIG['reg']
    )

    # 4. Huấn luyện (Training)
    print(f"[*] Bắt đầu training trong {CONFIG['epochs']} epochs...")
    history = model.fit(
        X_train_split, y_train_split,
        X_val=X_val_split, y_val=y_val_split,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        verbose=True
    )

    # 5. Đánh giá trên tập Test (SỬA LỖI TẠI ĐÂY)
    # Dùng predict để lấy nhãn dự đoán, sau đó so sánh với y_test
    y_pred_test = model.predict(X_test_feat)
    test_acc = np.mean(y_pred_test == y_test)
    print(f"[*] ĐỘ CHÍNH XÁC TRÊN TẬP TEST: {test_acc:.4f}")

    # 6. Lưu Model
    save_path = f"model_{feature_name.lower()}.npz"
    model.save(save_path)
    print(f"[*] Đã lưu model tại: {save_path}")

    return test_acc, history['train_loss'][-1]

def main():
    # 1. Load dữ liệu thô
    print("Đang tải dữ liệu MNIST...")
    X_train_raw, y_train, X_test_raw, y_test = load_mnist_data()
    
    if X_train_raw is None:
        return

    # Danh sách các loại đặc trưng cần thí nghiệm
    experiments = {
        "PIXEL": FeatureExtractor.get_pixel_features,
        "BLOCK_AVG": FeatureExtractor.get_block_features,
        # "SLIDING_BLOCK_AVG": FeatureExtractor.get_sliding_block_features,
        "EDGE": FeatureExtractor.get_edge_features  # Lưu ý: Canny chạy khá lâu
    }

    results = {}

    # Vòng lặp chạy thí nghiệm
    for name, func in experiments.items():
        acc, loss = train_and_evaluate(name, func, X_train_raw, y_train, X_test_raw, y_test)
        results[name] = {'Accuracy': acc, 'Final Loss': loss}

    # Tổng kết
    print(f"\n{'='*30}")
    print("TỔNG KẾT KẾT QUẢ")
    print(f"{'='*30}")
    print(f"{'Feature':<15} | {'Test Accuracy':<15} | {'Final Loss':<15}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<15} | {metrics['Accuracy']:.4f}          | {metrics['Final Loss']:.4f}")
    print(f"{'='*30}")

if __name__ == "__main__":
    main()
