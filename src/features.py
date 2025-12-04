import numpy as np
import cv2

class FeatureExtractor:
    """
    Handles feature extraction from raw MNIST images (28x28).
    Each method returns a 2D array of shape (N, n_features).
    """

    @staticmethod
    def _normalize(X):
        """
        Helper to normalize pixel values to [0, 1].
        """
        return X.astype(np.float32) / 255.0

    @staticmethod
    def get_pixel_features(images):
        """
        Feature 1: Normalized Pixel Intensity.
        Flattens the image from (28, 28) to (784,).
        
        Args:
            images: np.array of shape (N, 28, 28)
        Returns:
            np.array of shape (N, 784)
        """
        # Normalize first
        images_norm = FeatureExtractor._normalize(images)
        
        # Flatten: (N, 28, 28) -> (N, 784)
        n_samples = images.shape[0]
        return images_norm.reshape(n_samples, -1)

    @staticmethod
    def get_edge_features(images, min_val=100, max_val=200):
        """
        Feature 2: Edge Detection using Canny.
        Extracts structural edges and flattens result.
        
        Args:
            images: np.array of shape (N, 28, 28)
            min_val, max_val: Thresholds for Canny
        Returns:
            np.array of shape (N, 784)
        """
        n_samples = images.shape[0]
        edges_list = []

        # OpenCV Canny works on single channel 8-bit images
        # We iterate because cv2.Canny expects single image input usually
        for i in range(n_samples):
            # Ensure image is uint8 for OpenCV
            img_uint8 = images[i].astype(np.uint8)
            
            # Detect edges
            edge_img = cv2.Canny(img_uint8, min_val, max_val)
            
            # Normalize to 0-1
            edge_norm = edge_img.astype(np.float32) / 255.0
            edges_list.append(edge_norm.flatten())

        return np.array(edges_list)

    @staticmethod
    def get_block_features(images, block_size=(4, 4)):
        """
        Feature 3: Block Averaging (non-overlapping).
        Reduces 28x28 to (28/bh)x(28/bw) by averaging blocks.
        """
        images_norm = FeatureExtractor._normalize(images)
        n_samples, h, w = images_norm.shape
        bh, bw = block_size

        assert h % bh == 0 and w % bw == 0, \
            f"Image size {(h, w)} must be divisible by block size {block_size}"

        new_h, new_w = h // bh, w // bw
        reshaped = images_norm.reshape(n_samples, new_h, bh, new_w, bw)
        pooled = reshaped.mean(axis=(2, 4))
        return pooled.reshape(n_samples, -1)

    @staticmethod
    def get_sliding_block_features(
        images,
        window_size=(4, 4),
        stride=(2, 2),
        pooling="mean",
    ):
        """
        Feature X: Sliding-window block pooling (overlapping blocks).

        For each image, a sliding window of size (wh, ww) moves over the
        image with stride (sh, sw). Each window is pooled (mean or max),
        and the pooled values are flattened into a feature vector.

        Args:
            images: np.array of shape (N, H, W)
            window_size: (wh, ww), e.g. (4, 4)
            stride: int or (sh, sw), e.g. 2 or (2, 2)
            pooling: "mean", "max", or "flat"
                - "mean": average value in each window
                - "max": max value in each window
                - "flat": keep all pixels in each window (huge feature size)

        Returns:
            features: np.array of shape (N, n_features)
        """
        images_norm = FeatureExtractor._normalize(images)
        n_samples, H, W = images_norm.shape

        wh, ww = window_size
        if isinstance(stride, int):
            sh = sw = stride
        else:
            sh, sw = stride

        # Valid sliding-window positions (no padding)
        out_h = (H - wh) // sh + 1
        out_w = (W - ww) // sw + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Invalid window_size {window_size} and stride {stride} for image size {(H, W)}"
            )

        # Build a strided view: (N, out_h, out_w, wh, ww)
        sN, sH, sW = images_norm.strides
        patches = as_strided(
            images_norm,
            shape=(n_samples, out_h, out_w, wh, ww),
            strides=(sN, sH * sh, sW * sw, sH, sW),
            writeable=False,
        )

        if pooling == "mean":
            # (N, out_h, out_w)
            pooled = patches.mean(axis=(3, 4))
            return pooled.reshape(n_samples, -1)

        elif pooling == "max":
            pooled = patches.max(axis=(3, 4))
            return pooled.reshape(n_samples, -1)

        elif pooling == "flat":
            # Keep raw pixels from each window: very high-dimensional
            return patches.reshape(n_samples, -1)

        else:
            raise ValueError(f"Unknown pooling type: {pooling}")