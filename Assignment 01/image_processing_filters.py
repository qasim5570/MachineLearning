import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Read image
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('baboon.jpg', cv2.IMREAD_GRAYSCALE)

# ============ SMOOTHING/BLURRING FILTERS ============

# Gaussian Filter - Weighted average based on Gaussian distribution
# gaussian = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=1.0)
gaussian = ndimage.gaussian_filter(image, sigma=1.0)

# Median Filter - Replaces pixel with median of neighborhood (salt-and-pepper noise removal)
median = cv2.medianBlur(image, ksize=5)

# Bilateral Filter - Smooths while preserving edges (spatial + intensity weighting)
bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Mean/Box Filter - Simple uniform averaging (all weights equal)
mean = cv2.blur(image, ksize=(5, 5))

# ============ EDGE DETECTION FILTERS ============

# Sobel Filter - First derivative, gradient-based edge detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3)  # Vertical edges
sobel_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3)  # Horizontal edges
sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5,
                                 cv2.convertScaleAbs(sobel_y), 0.5, 0)

# Prewitt Filter - Similar to Sobel but uniform weights
prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
# prewitt_x = cv2.filter2D(image, cv2.CV_64F, prewitt_kernel_x)
prewitt_x = convolve2d(image, prewitt_kernel_x, mode='same', boundary='fill')
# prewitt_y = cv2.filter2D(image, cv2.CV_64F, prewitt_kernel_y)
prewitt_y = convolve2d(image, prewitt_kernel_y, mode='same', boundary='fill')
convolve2d(image, prewitt_kernel_x, mode='same', boundary='fill')
prewitt_combined = cv2.addWeighted(cv2.convertScaleAbs(prewitt_x), 0.5,
                                   cv2.convertScaleAbs(prewitt_y), 0.5, 0)

# Roberts Filter - Simple 2x2 diagonal gradient operator
roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
roberts_x = cv2.filter2D(image, cv2.CV_64F, roberts_kernel_x)
roberts_y = cv2.filter2D(image, cv2.CV_64F, roberts_kernel_y)
roberts_combined = cv2.addWeighted(cv2.convertScaleAbs(roberts_x), 0.5,
                                   cv2.convertScaleAbs(roberts_y), 0.5, 0)

# Scharr Filter - Improved Sobel with better rotation symmetry
scharr_x = cv2.Scharr(image, cv2.CV_64F, dx=1, dy=0)
scharr_y = cv2.Scharr(image, cv2.CV_64F, dx=0, dy=1)
scharr_combined = cv2.addWeighted(cv2.convertScaleAbs(scharr_x), 0.5,
                                  cv2.convertScaleAbs(scharr_y), 0.5, 0)

# Laplacian Filter - Second derivative, isotropic edge detection
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

# Canny Edge Detector - Multi-stage optimal edge detection
canny = cv2.Canny(image, threshold1=100, threshold2=200)

# ============ GABOR FILTER (Texture/Orientation) ============

def apply_gabor(image, ksize, sigma, theta, lambd, gamma, psi):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
    return cv2.filter2D(image, cv2.CV_8UC3, kernel)

# Gabor at 0 degrees (horizontal features)
gabor_0 = apply_gabor(image, ksize=21, sigma=5, theta=0, lambd=10, gamma=0.5, psi=0)
# Gabor at 90 degrees (vertical features)  
gabor_90 = apply_gabor(image, ksize=21, sigma=5, theta=np.pi/2, lambd=10, gamma=0.5, psi=0)
# Gabor at 45 degrees (diagonal features)
gabor_45 = apply_gabor(image, ksize=21, sigma=5, theta=np.pi/4, lambd=10, gamma=0.5, psi=0)

# ============ SHARPENING FILTER ============

# Unsharp Masking - Sharpen by subtracting blurred version from original
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 1.0)
unsharp_mask = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)

# ============ MORPHOLOGICAL OPERATIONS ============

# Define structuring element (kernel)
kernel = np.ones((5, 5), np.uint8)

# Erosion - Shrinks bright regions, removes small white noise
erosion = cv2.erode(image, kernel, iterations=1)

# Dilation - Expands bright regions, fills small holes
dilation = cv2.dilate(image, kernel, iterations=1)

# Opening - Erosion followed by Dilation (removes small objects/noise)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing - Dilation followed by Erosion (fills small holes/gaps)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# ============ DISPLAY ALL RESULTS ============

fig, axes = plt.subplots(4, 5, figsize=(20, 16))

# Row 1: Original + Smoothing filters
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original', fontsize=10)

axes[0, 1].imshow(gaussian, cmap='gray')
axes[0, 1].set_title('Gaussian Filter', fontsize=10)

axes[0, 2].imshow(median, cmap='gray')
axes[0, 2].set_title('Median Filter', fontsize=10)

axes[0, 3].imshow(bilateral, cmap='gray')
axes[0, 3].set_title('Bilateral Filter', fontsize=10)

axes[0, 4].imshow(mean, cmap='gray')
axes[0, 4].set_title('Mean/Box Filter', fontsize=10)

# Row 2: Edge detection - Gradient operators
axes[1, 0].imshow(sobel_combined, cmap='gray')
axes[1, 0].set_title('Sobel Filter', fontsize=10)

axes[1, 1].imshow(prewitt_combined, cmap='gray')
axes[1, 1].set_title('Prewitt Filter', fontsize=10)

axes[1, 2].imshow(roberts_combined, cmap='gray')
axes[1, 2].set_title('Roberts Filter', fontsize=10)

axes[1, 3].imshow(scharr_combined, cmap='gray')
axes[1, 3].set_title('Scharr Filter', fontsize=10)

axes[1, 4].imshow(laplacian, cmap='gray')
axes[1, 4].set_title('Laplacian Filter', fontsize=10)

# Row 3: Advanced edge detection + Gabor + Sharpening
axes[2, 0].imshow(canny, cmap='gray')
axes[2, 0].set_title('Canny Edge Detector', fontsize=10)

axes[2, 1].imshow(gabor_0, cmap='gray')
axes[2, 1].set_title('Gabor 0° (Horizontal)', fontsize=10)

axes[2, 2].imshow(gabor_90, cmap='gray')
axes[2, 2].set_title('Gabor 90° (Vertical)', fontsize=10)

axes[2, 3].imshow(gabor_45, cmap='gray')
axes[2, 3].set_title('Gabor 45° (Diagonal)', fontsize=10)

axes[2, 4].imshow(unsharp_mask, cmap='gray')
axes[2, 4].set_title('Unsharp Masking', fontsize=10)

# Row 4: Morphological operations
axes[3, 0].imshow(erosion, cmap='gray')
axes[3, 0].set_title('Erosion', fontsize=10)

axes[3, 1].imshow(dilation, cmap='gray')
axes[3, 1].set_title('Dilation', fontsize=10)

axes[3, 2].imshow(opening, cmap='gray')
axes[3, 2].set_title('Opening', fontsize=10)

axes[3, 3].imshow(closing, cmap='gray')
axes[3, 3].set_title('Closing', fontsize=10)

axes[3, 4].axis('off')  # Empty slot

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.suptitle('Classical Image Processing Filters', fontsize=16, y=1.02)
plt.show()