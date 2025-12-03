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
        Feature 3: Block Averaging (Dimensionality Reduction).
        Reduces 28x28 image to 7x7 by averaging 4x4 blocks.
        
        Args:
            images: np.array of shape (N, 28, 28)
            block_size: tuple (h, w) of block. For 28x28, (4,4) -> 7x7 output
        Returns:
            np.array of shape (N, 49)
        """
        # Normalize first
        images_norm = FeatureExtractor._normalize(images)
        
        n_samples, h, w = images_norm.shape
        bh, bw = block_size
        
        # Calculate new dimensions
        new_h, new_w = h // bh, w // bw
        
        # Reshape to (N, new_h, block_h, new_w, block_w)
        # This allows vectorizing the mean operation
        reshaped = images_norm.reshape(n_samples, new_h, bh, new_w, bw)
        
        # Mean over the blocks (axis 2 and 4)
        pooled = reshaped.mean(axis=(2, 4))
        
        # Flatten: (N, 7, 7) -> (N, 49)
        return pooled.reshape(n_samples, -1)