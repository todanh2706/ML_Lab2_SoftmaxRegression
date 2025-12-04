import numpy as np
import os
import time
from model import SoftmaxRegression
from features import FeatureExtractor

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
    """
    Hàm tải dữ liệu MNIST.
    Lưu ý: Nếu không được dùng keras để tải, bạn hãy thay thế bằng code đọc file binary thủ công.
    Ở đây dùng keras chỉ để lấy dữ liệu thô (raw data), không dùng cho model.
    """
    try:
        from tensorflow.keras.datasets import mnist
        (X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()
    except ImportError:
        print("Lỗi: Cần cài đặt tensorflow để tải data mẫu, hoặc tự load file mnist.npz")
        return None, None, None, None

    return X_train_raw, y_train, X_test_raw, y_test

def train_and_evaluate(feature_name, extract_func, X_raw, y, X_test_raw, y_test):
    print(f"\n{'='*10} ĐANG XỬ LÝ: {feature_name} {'='*10}")
    
    # 1. Trích xuất đặc trưng
    print(f"[*] Đang trích xuất đặc trưng (Feature Extraction)...")
    start_time = time.time()
    X_feat = extract_func(X_raw)
    X_test_feat = extract_func(X_test_raw)
    print(f"    - Thời gian trích xuất: {time.time() - start_time:.2f}s")
    print(f"    - Kích thước vector đặc trưng: {X_feat.shape[1]}")

    # 2. Chia tập Train/Validation thủ công
    num_train = int(X_feat.shape[0] * (1 - CONFIG['val_split']))
    
    X_train_split = X_feat[:num_train]
    y_train_split = y[:num_train]
    X_val_split = X_feat[num_train:]
    y_val_split = y[num_train:]

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