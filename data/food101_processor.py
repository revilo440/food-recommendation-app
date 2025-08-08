"""
Food-101 Dataset Preprocessing Pipeline
Generates embeddings for all Food-101 images and builds FAISS index
"""
import os
import json
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Any
import time
from pathlib import Path

from models.feature_extractor import FeatureExtractor
from models.similarity_engine import SimilarityEngine
from config import (
    FOOD101_IMAGES_PATH, FOOD101_EMBEDDINGS_PATH, FOOD_NAME_MAPPING_PATH,
    FAISS_INDEX_PATH, BATCH_SIZE
)


class Food101Processor:
    """
    Processor for Food-101 dataset preprocessing.
    Handles feature extraction, embedding storage, and FAISS index building.
    """
    
    def __init__(self):
        """Initialize the Food-101 processor."""
        self.feature_extractor = None
        self.similarity_engine = None
        
        # Ensure directories exist
        os.makedirs(FOOD101_EMBEDDINGS_PATH, exist_ok=True)
        
        logging.info("Food101Processor initialized")
    
    def _init_feature_extractor(self):
        """Initialize feature extractor if not already loaded."""
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
            logging.info("FeatureExtractor loaded")
    
    def validate_food101_dataset(self, dataset_path: str = FOOD101_IMAGES_PATH) -> Dict[str, Any]:
        """
        Validate Food-101 dataset structure and integrity.
        
        Args:
            dataset_path: Path to Food-101 images directory
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not os.path.exists(dataset_path):
                return {
                    "valid": False,
                    "error": f"Dataset path not found: {dataset_path}",
                    "stats": {}
                }
            
            # Get all food class directories
            food_classes = [d for d in os.listdir(dataset_path) 
                          if os.path.isdir(os.path.join(dataset_path, d))]
            
            stats = {
                "num_classes": len(food_classes),
                "classes": food_classes,
                "total_images": 0,
                "corrupted_images": [],
                "missing_classes": [],
                "class_stats": {}
            }
            
            # Expected Food-101 classes (should be 101)
            expected_classes = 101
            
            print(f"Validating {len(food_classes)} food classes...")
            
            for food_class in tqdm(food_classes, desc="Validating classes"):
                class_path = os.path.join(dataset_path, food_class)
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                valid_images = 0
                corrupted_images = []
                
                # Sample check a few images per class for corruption
                sample_images = image_files[:5] if len(image_files) > 5 else image_files
                
                for img_file in sample_images:
                    img_path = os.path.join(class_path, img_file)
                    try:
                        if self.feature_extractor is None:
                            self._init_feature_extractor()
                        
                        validation = self.feature_extractor.validate_image(img_path)
                        if validation["valid"]:
                            valid_images += 1
                        else:
                            corrupted_images.append(img_file)
                    except Exception as e:
                        corrupted_images.append(img_file)
                
                stats["class_stats"][food_class] = {
                    "total_images": len(image_files),
                    "sampled_valid": valid_images,
                    "sampled_corrupted": corrupted_images
                }
                stats["total_images"] += len(image_files)
                stats["corrupted_images"].extend([
                    os.path.join(food_class, img) for img in corrupted_images
                ])
            
            # Check if we have expected number of classes
            if len(food_classes) < expected_classes:
                stats["missing_classes"] = [f"Expected {expected_classes} classes, found {len(food_classes)}"]
            
            validation_result = {
                "valid": len(stats["corrupted_images"]) == 0 and len(food_classes) >= expected_classes,
                "error": None if len(stats["corrupted_images"]) == 0 else f"Found {len(stats['corrupted_images'])} corrupted images",
                "stats": stats
            }
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed: {str(e)}",
                "stats": {}
            }
    
    def create_food_name_mappings(self, dataset_path: str = FOOD101_IMAGES_PATH) -> Dict[str, str]:
        """
        Create mapping from snake_case food class names to display names.
        
        Args:
            dataset_path: Path to Food-101 images directory
            
        Returns:
            Dictionary mapping food_class -> display_name
        """
        try:
            food_classes = [d for d in os.listdir(dataset_path) 
                          if os.path.isdir(os.path.join(dataset_path, d))]
            
            # Create display name mappings
            food_mappings = {}
            for food_class in food_classes:
                # Convert snake_case to Title Case
                display_name = food_class.replace('_', ' ').title()
                food_mappings[food_class] = display_name
            
            # Save mappings to file
            with open(FOOD_NAME_MAPPING_PATH, 'w') as f:
                json.dump(food_mappings, f, indent=2)
            
            logging.info(f"Created {len(food_mappings)} food name mappings")
            return food_mappings
            
        except Exception as e:
            raise RuntimeError(f"Failed to create food name mappings: {str(e)}")
    
    def process_single_image(self, image_path: str, output_path: str) -> bool:
        """
        Process a single image and save embedding.
        
        Args:
            image_path: Path to input image
            output_path: Path to save embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.feature_extractor is None:
                self._init_feature_extractor()
            
            # Extract features
            features = self.feature_extractor.extract_features(image_path)
            
            # Save embedding
            np.save(output_path, features)
            
            return True
            
        except Exception as e:
            logging.warning(f"Failed to process {image_path}: {str(e)}")
            return False
    
    def process_food_class(self, food_class: str, dataset_path: str = FOOD101_IMAGES_PATH, 
                          max_images_per_class: int = None) -> Tuple[int, int]:
        """
        Process images in a food class (optionally limited to a subset).
        
        Args:
            food_class: Name of the food class
            dataset_path: Path to Food-101 images directory
            max_images_per_class: Maximum number of images to process per class (None for all)
            
        Returns:
            Tuple of (successful_count, total_count)
        """
        class_path = os.path.join(dataset_path, food_class)
        class_embedding_path = os.path.join(FOOD101_EMBEDDINGS_PATH, food_class)
        
        # Create class embedding directory
        os.makedirs(class_embedding_path, exist_ok=True)
        
        # Get all image files
        all_image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit number of images if specified
        if max_images_per_class is not None:
            image_files = all_image_files[:max_images_per_class]
        else:
            image_files = all_image_files
        
        successful = 0
        
        for img_file in tqdm(image_files, desc=f"Processing {food_class}", leave=False):
            img_path = os.path.join(class_path, img_file)
            
            # Create embedding filename
            img_name = os.path.splitext(img_file)[0]
            embedding_file = f"{food_class}_{img_name}.npy"
            embedding_path = os.path.join(class_embedding_path, embedding_file)
            
            # Skip if embedding already exists
            if os.path.exists(embedding_path):
                successful += 1
                continue
            
            # Process image
            if self.process_single_image(img_path, embedding_path):
                successful += 1
        
        return successful, len(image_files)
    
    def setup_food101(self, dataset_path: str = FOOD101_IMAGES_PATH, 
                     rebuild_embeddings: bool = False, max_images_per_class: int = None) -> Dict[str, Any]:
        """
        Main setup function for Food-101 dataset preprocessing.
        
        Args:
            dataset_path: Path to Food-101 images directory
            rebuild_embeddings: Whether to rebuild existing embeddings
            max_images_per_class: Maximum number of images to process per class (None for all)
            
        Returns:
            Dictionary with processing results
        """
        try:
            print("🍔 Starting Food-101 dataset preprocessing...")
            start_time = time.time()
            
            # Initialize feature extractor
            self._init_feature_extractor()
            
            # Validate dataset
            print("📋 Validating Food-101 dataset...")
            validation = self.validate_food101_dataset(dataset_path)
            
            if not validation["valid"]:
                print(f"Dataset validation failed: {validation['error']}")
                return {"success": False, "error": validation["error"]}
            
            stats = validation["stats"]
            print(f"Dataset validated: {stats['num_classes']} classes, {stats['total_images']} total images")
            
            # Create food name mappings
            print("Creating food name mappings...")
            self.create_food_name_mappings(dataset_path)
            
            # Clear existing embeddings if rebuilding
            if rebuild_embeddings and os.path.exists(FOOD101_EMBEDDINGS_PATH):
                print("🧹 Clearing existing embeddings...")
                import shutil
                shutil.rmtree(FOOD101_EMBEDDINGS_PATH)
                os.makedirs(FOOD101_EMBEDDINGS_PATH, exist_ok=True)
            
            # Process all food classes
            subset_msg = f" (max {max_images_per_class} per class)" if max_images_per_class else " (all images)"
            print(f"Processing food classes{subset_msg}...")
            food_classes = stats["classes"]
            total_successful = 0
            total_images = 0
            
            processing_stats = {}
            
            for food_class in tqdm(food_classes, desc="Processing classes"):
                successful, total = self.process_food_class(food_class, dataset_path, max_images_per_class)
                processing_stats[food_class] = {
                    "successful": successful,
                    "total": total,
                    "success_rate": successful / total if total > 0 else 0
                }
                total_successful += successful
                total_images += total
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "processing_time_seconds": processing_time,
                "total_images_processed": total_successful,
                "total_images": total_images,
                "success_rate": total_successful / total_images if total_images > 0 else 0,
                "num_classes": len(food_classes),
                "embeddings_path": FOOD101_EMBEDDINGS_PATH,
                "processing_stats": processing_stats
            }
            
            print("Food-101 preprocessing completed!")
            print(f"Processed: {total_successful}/{total_images} images ({result['success_rate']:.1%})")
            print(f"Time: {processing_time:.1f} seconds")
            print(f"Embeddings saved to: {FOOD101_EMBEDDINGS_PATH}")
            
            return result
            
        except Exception as e:
            error_msg = f"Food-101 setup failed: {str(e)}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def build_faiss_index(self, embeddings_path: str = FOOD101_EMBEDDINGS_PATH) -> Dict[str, Any]:
        """
        Build FAISS index from generated embeddings.
        
        Args:
            embeddings_path: Path to directory containing embeddings
            
        Returns:
            Dictionary with index building results
        """
        try:
            print("🔍 Building FAISS index from embeddings...")
            start_time = time.time()
            
            # Initialize similarity engine
            if self.similarity_engine is None:
                self.similarity_engine = SimilarityEngine(embeddings_path)
            
            # Collect all embedding files
            embedding_files = []
            embeddings_list = []
            
            print("📁 Collecting embedding files...")
            for root, dirs, files in os.walk(embeddings_path):
                for file in files:
                    if file.endswith('.npy'):
                        embedding_path = os.path.join(root, file)
                        try:
                            embedding = np.load(embedding_path)
                            if embedding.shape == (2048,):  # Validate embedding shape
                                embedding_files.append(file)
                                embeddings_list.append(embedding)
                        except Exception as e:
                            logging.warning(f"Failed to load embedding {embedding_path}: {str(e)}")
            
            if not embeddings_list:
                raise RuntimeError("No valid embeddings found")
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings_list)
            print(f"Loaded {len(embeddings_list)} embeddings")
            
            # Build FAISS index
            print("Building FAISS index...")
            index = self.similarity_engine.build_faiss_index(embeddings_array, embedding_files)
            self.similarity_engine.index = index  # Assign the index to the engine
            
            # Save index
            print("Saving FAISS index...")
            self.similarity_engine.save_faiss_index()
            
            build_time = time.time() - start_time
            
            result = {
                "success": True,
                "num_embeddings": len(embeddings_list),
                "index_path": FAISS_INDEX_PATH,
                "build_time_seconds": build_time,
                "index_size_mb": os.path.getsize(FAISS_INDEX_PATH) / (1024 * 1024) if os.path.exists(FAISS_INDEX_PATH) else 0
            }
            
            print(f"FAISS index built successfully!")
            print(f"Indexed: {len(embeddings_list)} embeddings")
            print(f"Time: {build_time:.1f} seconds")
            print(f"Index saved to: {FAISS_INDEX_PATH}")
            print(f"Index size: {result['index_size_mb']:.1f} MB")
            
            return result
            
        except Exception as e:
            error_msg = f"FAISS index building failed: {str(e)}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Food-101 Dataset Processor")
    parser.add_argument("--validate-dataset", action="store_true", 
                       help="Validate Food-101 dataset only")
    parser.add_argument("--rebuild-embeddings", action="store_true",
                       help="Rebuild all embeddings from scratch")
    parser.add_argument("--build-index-only", action="store_true",
                       help="Build FAISS index only (from existing embeddings)")
    parser.add_argument("--full-setup", action="store_true",
                       help="Run full setup: embeddings + index")
    parser.add_argument("--max-images-per-class", type=int, default=10,
                       help="Maximum number of images to process per class (default: 10)")
    
    args = parser.parse_args()
    
    processor = Food101Processor()
    
    if args.validate_dataset:
        print("Validating Food-101 dataset...")
        result = processor.validate_food101_dataset()
        if result["valid"]:
            print("Dataset validation passed")
            print(f"{result['stats']['num_classes']} classes, {result['stats']['total_images']} images")
        else:
            print(f"Dataset validation failed: {result['error']}")
    
    elif args.build_index_only:
        result = processor.build_faiss_index()
        if not result["success"]:
            print(f"Index building failed: {result['error']}")
    
    elif args.full_setup or args.rebuild_embeddings:
        # Setup embeddings
        result = processor.setup_food101(rebuild_embeddings=args.rebuild_embeddings, 
                                       max_images_per_class=args.max_images_per_class)
        if result["success"]:
            # Build index
            index_result = processor.build_faiss_index()
            if not index_result["success"]:
                print(f"Index building failed: {index_result['error']}")
        else:
            print(f"Embedding generation failed: {result['error']}")
    
    else:
        print("Please specify an action: --validate-dataset, --build-index-only, or --full-setup")


if __name__ == "__main__":
    main()