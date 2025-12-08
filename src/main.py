import numpy as np
import os
import time
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from model import SoftmaxRegression           # Đổi lại nếu file tên khác
from features import FeatureExtractor
from data import train_val_split, load_mnist_npz
from dataLoader import MnistDataloader


# Cấu hình Hyperparameters
CONFIG = {
    "lr": 0.1,
    "reg": 1e-4,
    "epochs": 20,
    "batch_size": 128,
    "n_classes": 10,
    "val_split": 0.1,  # 10% tập train dùng để validate
}


def load_mnist_data():
    """
    Load MNIST data from local 'data' folder using MnistDataloader.
    Requires: train-images.idx3-ubyte, train-labels.idx1-ubyte, etc.
    """
    # 1. Đường dẫn đến thư mục chứa data (nằm cùng cấp với main.py hoặc trong folder data)
    # Giả sử bạn để trong folder 'data' nằm cùng cấp với main.py
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # Định nghĩa tên file chuẩn
    files = {
        "train_images": os.path.join(data_dir, "train-images.idx3-ubyte"),
        "train_labels": os.path.join(data_dir, "train-labels.idx1-ubyte"),
        "test_images":  os.path.join(data_dir, "t10k-images.idx3-ubyte"),
        "test_labels":  os.path.join(data_dir, "t10k-labels.idx1-ubyte"),
    }

    # 2. Kiểm tra file tồn tại
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"Lỗi: Không tìm thấy file {name} tại đường dẫn: {path}")
            print("Vui lòng tải dataset MNIST (định dạng idx) và đặt vào thư mục 'data'.")
            return None, None, None, None

    print(f"Đang tải dữ liệu từ: {data_dir} ...")

    # 3. Sử dụng dataLoader.py có sẵn để đọc
    mnist_dataloader = MnistDataloader(
        files["train_images"],
        files["train_labels"],
        files["test_images"],
        files["test_labels"],
    )
    
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # 4. Chuyển đổi sang Numpy array (đúng định dạng uint8 để xử lý ảnh)
    return (
        np.array(x_train, dtype=np.uint8),
        np.array(y_train, dtype=np.uint8),
        np.array(x_test, dtype=np.uint8),
        np.array(y_test, dtype=np.uint8),
    )

def plot_metric_bars(results, metric_key, ylabel, title, filename):
    """
    Vẽ bar chart cho 1 metric: Accuracy, Macro F1, Macro Precision, Macro Recall.
    results: dict { feature: {metric: value, ...} }
    """
    features = list(results.keys())
    values = [results[f][metric_key] for f in features]

    plt.figure(figsize=(7,5))
    plt.bar(features, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.ylim(0.8, 1)

    for i, v in enumerate(values):
        plt.text(i, v + 0.001, f"{v:.4f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[*] Đã lưu biểu đồ: {filename}")


def plot_multi_metric(results, filename="comparison_all_metrics.png"):
    """
    Vẽ một biểu đồ cột nhóm, so sánh Accuracy – Precision – Recall – F1.
    """
    features = list(results.keys())

    acc = [results[f]["Accuracy"] for f in features]
    f1  = [results[f]["Macro_F1"] for f in features]
    prec = [results[f]["Macro_Precision"] for f in features]
    rec = [results[f]["Macro_Recall"] for f in features]

    x = np.arange(len(features))
    width = 0.2

    plt.figure(figsize=(10,6))
    plt.bar(x - 1.5*width, acc, width, label="Accuracy")
    plt.bar(x - 0.5*width, prec, width, label="Precision")
    plt.bar(x + 0.5*width, rec, width, label="Recall")
    plt.bar(x + 1.5*width, f1, width, label="F1-score")

    plt.xticks(x, features)
    plt.title("So sánh tổng quát giữa các Feature")
    plt.ylabel("Score")
    plt.ylim(0.8, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()

    # In giá trị lên đầu cột
    def annotate(values, shift):
        for xx, val in zip(x + shift, values):
            plt.text(xx, val + 0.003, f"{val:.3f}", ha='center', fontsize=8)

    annotate(acc, -1.5*width)
    annotate(prec, -0.5*width)
    annotate(rec, 0.5*width)
    annotate(f1, 1.5*width)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[*] Đã lưu biểu đồ: {filename}")


def plot_confusion_matrix(cm, classes, feature_name, normalize=False):
    """
    Vẽ và lưu confusion matrix thành file png.
    cm: confusion_matrix (ndarray)
    classes: list các nhãn (vd [0,1,2,...,9])
    feature_name: tên loại feature (PIXEL, BLOCK_AVG, ...)
    normalize: nếu True thì normalize theo hàng.
    """
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_sum[cm_sum == 0] = 1
        cm = cm.astype("float") / cm_sum

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion matrix - {feature_name}")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    # Lưu file
    fname = f"confusion_matrix_{feature_name.lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"    -> Đã lưu confusion matrix tại: {fname}")


def evaluate_classification(y_true, y_pred, n_classes, feature_name):
    """
    Tính các metric classification + in báo cáo.
    Trả về dict chứa metric summary để so sánh giữa các feature.
    """
    labels = np.arange(n_classes)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 theo từng lớp
    precision_per_cls, recall_per_cls, f1_per_cls, support_per_cls = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
    )

    # Macro-average (trung bình đều các lớp)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n================= ĐÁNH GIÁ CHI TIẾT =================")
    print(f"Feature: {feature_name}")
    print(f"- Accuracy: {acc:.4f}")
    print(f"- Macro Precision: {macro_precision:.4f}")
    print(f"- Macro Recall:    {macro_recall:.4f}")
    print(f"- Macro F1-score:  {macro_f1:.4f}")

    print("\n--- Precision / Recall / F1 theo từng lớp ---")
    for idx, cls in enumerate(labels):
        print(
            f"  Lớp {cls}: "
            f"Precision={precision_per_cls[idx]:.4f}, "
            f"Recall={recall_per_cls[idx]:.4f}, "
            f"F1={f1_per_cls[idx]:.4f}, "
            f"Support={support_per_cls[idx]}"
        )

    print("\n--- Classification report (sklearn) ---")
    print(
        classification_report(
            y_true, y_pred, labels=labels, digits=4, zero_division=0
        )
    )

    # Vẽ + lưu confusion matrix (raw và normalized)
    print("\n--- Confusion matrix (raw) ---")
    plot_confusion_matrix(cm, classes=labels, feature_name=feature_name, normalize=False)
    print("\n--- Confusion matrix (normalized theo hàng) ---")
    plot_confusion_matrix(
        cm, classes=labels, feature_name=feature_name + "_norm", normalize=True
    )

    # Dict để dùng sau ở phần tổng kết / so sánh
    metrics_summary = {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }
    return metrics_summary


def train_and_evaluate(feature_name, extract_func, X_raw, y, X_test_raw, y_test):
    print(f"\n{'=' * 10} ĐANG XỬ LÝ: {feature_name} {'=' * 10}")

    # 1. Trích xuất đặc trưng
    print(f"[*] Đang trích xuất đặc trưng (Feature Extraction)...")
    start_time = time.time()
    X_feat = extract_func(X_raw)
    X_test_feat = extract_func(X_test_raw)
    print(f"    - Thời gian trích xuất: {time.time() - start_time:.2f}s")
    print(f"    - Kích thước vector đặc trưng: {X_feat.shape[1]}")

    # 2. Chia tập train/val
    X_train_split, y_train_split, X_val_split, y_val_split = train_val_split(
        X_feat, y, val_ratio=CONFIG["val_split"], shuffle=True, seed=42
    )

    # 3. Khởi tạo Model
    model = SoftmaxRegression(
        n_features=X_feat.shape[1],
        n_classes=CONFIG["n_classes"],
        lr=CONFIG["lr"],
        reg=CONFIG["reg"],
    )

    # 4. Huấn luyện (Training)
    print(f"[*] Bắt đầu training trong {CONFIG['epochs']} epochs...")
    history = model.fit(
        X_train_split,
        y_train_split,
        X_val=X_val_split,
        y_val=y_val_split,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        verbose=True,
    )

    # 5. Đánh giá trên tập Test với các metric đầy đủ
    y_pred_test = model.predict(X_test_feat)
    test_metrics = evaluate_classification(
        y_true=y_test,
        y_pred=y_pred_test,
        n_classes=CONFIG["n_classes"],
        feature_name=feature_name,
    )

    # 6. Lưu Model
    save_path = f"model_{feature_name.lower()}.npz"
    model.save(save_path)
    print(f"[*] Đã lưu model tại: {save_path}")

    # Trả về metrics + final train loss (để tiện log)
    return test_metrics, history["train_loss"][-1]


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
        "EDGE": FeatureExtractor.get_edge_features,  # Lưu ý: Canny chạy khá lâu
    }

    results = {}

    # Vòng lặp chạy thí nghiệm
    for name, func in experiments.items():
        metrics, final_loss = train_and_evaluate(
            name, func, X_train_raw, y_train, X_test_raw, y_test
        )
        results[name] = {
            "Accuracy": metrics["accuracy"],
            "Macro_F1": metrics["macro_f1"],
            "Macro_Precision": metrics["macro_precision"],
            "Macro_Recall": metrics["macro_recall"],
            "Final_Loss": final_loss,
        }

    # Tổng kết
    print(f"\n{'=' * 40}")
    print("TỔNG KẾT KẾT QUẢ GIỮA CÁC FEATURE")
    print(f"{'=' * 40}")
    header = (
        f"{'Feature':<15} | {'Acc':>7} | {'Macro F1':>9} | "
        f"{'Macro P':>9} | {'Macro R':>9} | {'Final Loss':>11}"
    )
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<15} | "
            f"{metrics['Accuracy']:.4f} | "
            f"{metrics['Macro_F1']:.4f} | "
            f"{metrics['Macro_Precision']:.4f} | "
            f"{metrics['Macro_Recall']:.4f} | "
            f"{metrics['Final_Loss']:.4f}"
        )
    print(f"{'=' * 40}")

    # ======= VẼ BIỂU ĐỒ SO SÁNH =======
    print("\nĐang vẽ biểu đồ so sánh...")

    plot_metric_bars(
        results, "Accuracy",
        ylabel="Accuracy",
        title="So sánh Accuracy giữa các Feature",
        filename="chart_accuracy.png"
    )

    plot_metric_bars(
        results, "Macro_F1",
        ylabel="Macro F1-score",
        title="So sánh Macro F1 giữa các Feature",
        filename="chart_macro_f1.png"
    )

    plot_metric_bars(
        results, "Macro_Precision",
        ylabel="Macro Precision",
        title="So sánh Precision giữa các Feature",
        filename="chart_precision.png"
    )

    plot_metric_bars(
        results, "Macro_Recall",
        ylabel="Macro Recall",
        title="So sánh Recall giữa các Feature",
        filename="chart_recall.png"
    )

    # Biểu đồ tổng hợp
    plot_multi_metric(results)


if __name__ == "__main__":
    main()
