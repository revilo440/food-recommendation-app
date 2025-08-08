"""
Feature Extractor using ResNet50 for food image embeddings
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os
from typing import Union, Dict, Any
import logging

from config import MODEL_NAME, EMBEDDING_SIZE, IMAGE_SIZE, SUPPORTED_FORMATS, MAX_FILE_SIZE_MB


class FeatureExtractor:
    """
    Feature extractor using pre-trained ResNet50 for food image embeddings.
    Extracts 2048-dimensional feature vectors from input images.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the feature extractor with a pre-trained ResNet50 model.
        
        Args:
            model_name: Name of the model to use (default: 'resnet50')
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        
        self._load_model()
        self._setup_transforms()
        
        logging.info(f"FeatureExtractor initialized with {model_name} on {self.device}")
    
    def _load_model(self):
        """Load and prepare the pre-trained ResNet50 model."""
        try:
            # Load pre-trained ResNet50
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            
            # Remove the final classification layer to get features
            # ResNet50 outputs 2048-dimensional features before the final FC layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
            # Set to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            # Freeze all parameters since we're using it as a feature extractor
            for param in self.model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name} model: {str(e)}")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms for ImageNet normalization."""
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]   # ImageNet std
            )
        ])
    
    def validate_image(self, image_path: str) -> Dict[str, Any]:
        """
        Validate uploaded image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with validation result: {"valid": bool, "error": str or None}
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return {"valid": False, "error": "File does not exist"}
            
            # Check file size
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return {"valid": False, "error": f"File size ({file_size_mb:.1f}MB) exceeds {MAX_FILE_SIZE_MB}MB limit"}
            
            # Check file extension
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in SUPPORTED_FORMATS:
                return {"valid": False, "error": f"Unsupported format {file_ext}. Supported: {SUPPORTED_FORMATS}"}
            
            # Try to open and validate the image
            with Image.open(image_path) as img:
                # Check if image can be loaded
                img.verify()
                
            # Reopen for size check (verify() closes the image)
            with Image.open(image_path) as img:
                width, height = img.size
                if width < IMAGE_SIZE[0] or height < IMAGE_SIZE[1]:
                    return {"valid": False, "error": f"Image too small ({width}x{height}). Minimum: {IMAGE_SIZE}"}
                
            return {"valid": True, "error": None}
            
        except Exception as e:
            error_msg = str(e).lower()
            if "corrupted" in error_msg or "truncated" in error_msg:
                return {"valid": False, "error": "Image file appears to be corrupted"}
            elif "format" in error_msg or "cannot identify" in error_msg:
                return {"valid": False, "error": "Unsupported or invalid image format"}
            else:
                return {"valid": False, "error": f"Image validation failed: {str(e)}"}
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for ResNet50 inference.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            Preprocessed image tensor ready for model inference
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image)
            
            # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed: {str(e)}")
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract feature vector from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            2048-dimensional feature vector as numpy array
            
        Raises:
            RuntimeError: If feature extraction fails
        """
        try:
            # Validate image first
            validation = self.validate_image(image_path)
            if not validation["valid"]:
                raise ValueError(f"Image validation failed: {validation['error']}")
            
            # Preprocess image
            tensor = self.preprocess_image(image_path)
            
            # Extract features
            with torch.no_grad():
                features = self.model(tensor)
                
                # Remove spatial dimensions (global average pooling is already applied)
                # ResNet50 outputs shape: (batch_size, 2048, 1, 1)
                features = features.squeeze()
                
                # Convert to numpy
                features = features.cpu().numpy()
                
                # Ensure we have the right shape
                if features.shape != (EMBEDDING_SIZE,):
                    raise RuntimeError(f"Expected feature shape {(EMBEDDING_SIZE,)}, got {features.shape}")
                
                return features
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise e  # Re-raise validation errors as-is
            else:
                raise RuntimeError(f"Feature extraction failed for {image_path}: {str(e)}")
    
    def extract_features_batch(self, image_paths: list) -> np.ndarray:
        """
        Extract features from multiple images in batch.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Array of shape (num_images, 2048) containing feature vectors
        """
        features_list = []
        
        for image_path in image_paths:
            try:
                features = self.extract_features(image_path)
                features_list.append(features)
            except Exception as e:
                logging.warning(f"Failed to extract features from {image_path}: {str(e)}")
                continue
        
        if not features_list:
            raise RuntimeError("No valid features extracted from batch")
        
        return np.array(features_list)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_size": EMBEDDING_SIZE,
            "device": str(self.device),
            "image_size": IMAGE_SIZE,
            "supported_formats": SUPPORTED_FORMATS,
            "max_file_size_mb": MAX_FILE_SIZE_MB
        }


def test_feature_extractor(image_path: str = None):
    """
    Test function for the FeatureExtractor class.
    
    Args:
        image_path: Path to test image (optional)
    """
    print("Testing FeatureExtractor...")
    
    try:
        # Initialize extractor
        extractor = FeatureExtractor()
        print(f"✓ FeatureExtractor initialized: {extractor.get_model_info()}")
        
        # Test with provided image or find a sample from Food-101
        if image_path is None:
            # Use first apple pie image from Food-101 dataset
            sample_path = "food-101/images/apple_pie"
            if os.path.exists(sample_path):
                sample_images = [f for f in os.listdir(sample_path) if f.endswith('.jpg')]
                if sample_images:
                    image_path = os.path.join(sample_path, sample_images[0])
        
        if image_path and os.path.exists(image_path):
            # Validate image
            validation = extractor.validate_image(image_path)
            print(f"✓ Image validation: {validation}")
            
            if validation["valid"]:
                # Extract features
                features = extractor.extract_features(image_path)
                print(f"✓ Features extracted: shape={features.shape}, dtype={features.dtype}")
                print(f"  Feature stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")
            else:
                print(f"✗ Image validation failed: {validation['error']}")
        else:
            print("✗ No test image available")
        
        print("FeatureExtractor test completed successfully!")
        
    except Exception as e:
        print(f"✗ FeatureExtractor test failed: {str(e)}")


if __name__ == "__main__":
    test_feature_extractor()