import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, segmentation, measure, exposure
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy import ndimage
from readlif.reader import LifFile
import tifffile
import oiffile
import cv2
import pywt
import pywt.data


class BiofilmProcessor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.image_data = None
        self.processed_data = None
        
    def load_image(self):
        """Load different microscopy file formats"""
        ext = self.file_path.lower().split('.')[-1]
        
        if ext == 'lif':
            reader = LifFile(self.file_path)
            # Get first image and first timepoint for now
            image = next(reader.get_iter_image())
            self.image_data = np.array(next(image.get_iter_t()))
        elif ext == 'lsm':
            self.image_data = tifffile.imread(self.file_path)
        elif ext == 'oif':
            self.image_data = oiffile.imread(self.file_path)
        else:
            # For standard formats (tiff, jpg, png)
            self.image_data = io.imread(self.file_path)
            
        return self.image_data

    def denoise_image(self, strength=0.1):
        """Depth-aware denoising with bilateral filter"""
        if len(self.image_data.shape) == 3:  # 3D stack
            self.processed_data = np.zeros_like(self.image_data)
            for z in range(self.image_data.shape[0]):
                # Increase denoising strength with depth
                depth_factor = 1 + (z / self.image_data.shape[0]) * strength
                self.processed_data[z] = cv2.bilateralFilter(
                    self.image_data[z].astype(np.float32),
                    d=5,
                    sigmaColor=50 * depth_factor,
                    sigmaSpace=50
                )
        else:
            self.processed_data = cv2.bilateralFilter(
                self.image_data.astype(np.float32), d=5, sigmaColor=50, sigmaSpace=50
            )
            
    def sharpen_image(self):
        """Sharpen the processed image using Unsharp Mask"""
        if self.processed_data is None:
            self.processed_data = self.image_data

        self.sharpened_data = np.zeros_like(self.processed_data)
        if len(self.processed_data.shape) == 3:  # 3D stack
            for z in range(self.processed_data.shape[0]):
                blurred = cv2.GaussianBlur(self.processed_data[z], (5, 5), 10)
                self.sharpened_data[z] = cv2.addWeighted(self.processed_data[z], 1.5, blurred, -0.5, 0)
        else:
            blurred = cv2.GaussianBlur(self.processed_data, (5, 5), 10)
            self.sharpened_data = cv2.addWeighted(self.processed_data, 1.5, blurred, -0.5, 0)

        return self.sharpened_data
    
    def fourier_based_sharpening(self):
            # Handle RGB images
            if len(self.processed_data.shape) == 3:
                # Process each channel separately
                sharpened = np.zeros_like(self.processed_data)
                for channel in range(self.processed_data.shape[2]):
                    # Get 2D slice
                    channel_data = self.processed_data[:,:,channel]
                    
                    # Fourier transform
                    f_transform = np.fft.fft2(channel_data)
                    f_shift = np.fft.fftshift(f_transform)
    
                    # Create high-pass filter mask
                    rows, cols = channel_data.shape
                    crow, ccol = rows // 2, cols // 2
                    mask = np.ones((rows, cols), dtype=np.float32)
                    r = 30  # Kernel size, high frequency threshold
                    center = (crow, ccol)
                    for i in range(rows):
                        for j in range(cols):
                            if np.sqrt((i - center[0])**2 + (j - center[1])**2) < r:
                                mask[i, j] = 0
    
                    # Apply mask
                    f_filtered = f_shift * mask
                    f_ishift = np.fft.ifftshift(f_filtered)
                    channel_filtered = np.abs(np.fft.ifft2(f_ishift))
                    
                    # Store result
                    sharpened[:,:,channel] = channel_filtered
                    
                # Normalize result
                self.enhanced_data = exposure.rescale_intensity(sharpened)


            
    def adaptive_threshold(self, block_size=35, offset=10):
        """Adaptive thresholding for segmentation"""
        if self.processed_data is None:
            self.processed_data = self.image_data
            
        # Normalize image
        normalized = exposure.rescale_intensity(self.processed_data)
        
        # Apply adaptive threshold
        thresh = filters.threshold_local(normalized, block_size, offset=offset)
        self.binary_mask = normalized > thresh
        
        return self.binary_mask

    def segment_objects(self, min_size=100):
        """Segment and label objects"""
        # Convert to grayscale if image is RGB
        if len(self.binary_mask.shape) == 3:
            binary_mask_2d = self.binary_mask[:,:,0]  # Take first channel
        else:
            binary_mask_2d = self.binary_mask

        # Use watershed segmentation for overlapping objects
        distance = ndimage.distance_transform_edt(binary_mask_2d)
        
        # Create markers using peak_local_max
        local_maxi = feature.peak_local_max(
            distance,
            min_distance=20,
            labels=binary_mask_2d
        )
        
        # Create markers from coordinates
        markers = np.zeros_like(binary_mask_2d, dtype=int)
        for coords in local_maxi:
            markers[coords[0], coords[1]] = 1
        markers = measure.label(markers)
        
        # Watershed segmentation
        self.labels = segmentation.watershed(-distance, markers, mask=binary_mask_2d)
        
        # Filter small objects
        self.labels = measure.label(self.labels)
        
        if len(self.processed_data.shape) == 3:
            # For RGB images, use first channel for intensity measurements
            intensity_image = self.processed_data[:,:,0]
        else:
            intensity_image = self.processed_data
        
        self.props = measure.regionprops(self.labels, intensity_image=intensity_image)
        
        # Remove small objects
        for prop in self.props:
            if prop.area < min_size:
                self.labels[self.labels == prop.label] = 0
                
        return self.labels

    def analyze_objects(self):
        """Extract features from segmented objects"""
        features = []
        for prop in self.props:
            features.append({
                'area': prop.area,
                'perimeter': prop.perimeter,
                'eccentricity': prop.eccentricity,
                'mean_intensity': prop.mean_intensity,
                'texture_variance': np.var(prop.intensity_image),
                'circularity': 4 * np.pi * prop.area / (prop.perimeter ** 2)
            })
        return features

    def visualize_results(self):
        """Display processing results including segmentation"""
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # 5 subplots
        
        # Original Image
        axes[0].imshow(self.image_data, cmap='gray')
        axes[0].set_title('Original Image')
        
        # Denoised Image
        axes[1].imshow(self.processed_data, cmap='gray')
        axes[1].set_title('Denoised Image')
        
        # Sharpened Image
        axes[2].imshow(self.sharpened_data, cmap='gray')
        axes[2].set_title('Sharpened Image')
        
        # Enhanced Contrast Image
        axes[3].imshow(self.enhanced_data, cmap='gray')
        axes[3].set_title('Fourier-based Sharpening')
        
        # Segmentation Results
        if hasattr(self, 'labels'):  # Check if segmentation is available
            from skimage.color import label2rgb  # Import for visualization
            labeled_image = label2rgb(self.labels, bg_label=0)  # Convert labels to RGB
            axes[4].imshow(labeled_image)
            axes[4].set_title('Segmentation Results')
        else:
            axes[4].text(0.5, 0.5, 'No Segmentation', fontsize=12, ha='center', va='center')
            axes[4].set_title('Segmentation Results')
        
        # Remove axes
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

processor = BiofilmProcessor("d4.jpg")
processor.load_image()
processor.denoise_image()
processor.sharpen_image()
processor.fourier_based_sharpening()
processor.adaptive_threshold()
processor.segment_objects()
features = processor.analyze_objects()
processor.visualize_results()



