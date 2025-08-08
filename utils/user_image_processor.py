"""
User Image Processing Pipeline
Handles user image uploads, feature extraction, and storage
"""
import os
import uuid
import shutil
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import logging

from models.feature_extractor import FeatureExtractor
from models.similarity_engine import SimilarityEngine
from data.database_manager import DatabaseManager
from config import (
    USER_IMAGES_PATH, USER_EMBEDDINGS_PATH, APP_DATA_PATH,
    FAISS_INDEX_PATH, SUPPORTED_FORMATS, MAX_FILE_SIZE_MB
)


class UserImageProcessor:
    """
    Processor for handling user uploaded food images.
    Handles validation, feature extraction, storage, and recommendations.
    """
    
    def __init__(self):
        """Initialize the user image processor."""
        self.feature_extractor = None
        self.similarity_engine = None
        self.database_manager = None
        
        # Ensure directories exist
        os.makedirs(USER_IMAGES_PATH, exist_ok=True)
        os.makedirs(USER_EMBEDDINGS_PATH, exist_ok=True)
        
        logging.info("UserImageProcessor initialized")
    
    def _init_components(self):
        """Initialize ML components if not already loaded."""
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
        
        if self.similarity_engine is None:
            self.similarity_engine = SimilarityEngine()
            # Load FAISS index if available
            if os.path.exists(FAISS_INDEX_PATH):
                self.similarity_engine.load_faiss_index()
                logging.info("Loaded FAISS index for recommendations")
            else:
                logging.warning("FAISS index not found. Run Food-101 preprocessing first.")
        
        if self.database_manager is None:
            self.database_manager = DatabaseManager()
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """
        Generate unique filename for user uploaded image.
        
        Args:
            original_filename: Original filename from upload
            
        Returns:
            Unique filename with UUID prefix
        """
        # Extract file extension
        _, ext = os.path.splitext(original_filename)
        if not ext:
            ext = '.jpg'  # Default extension
        
        # Generate unique filename with timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        unique_filename = f"user_{timestamp}_{unique_id}{ext}"
        
        return unique_filename
    
    def validate_and_save_image(self, uploaded_file, original_filename: str) -> Dict[str, Any]:
        """
        Validate and save user uploaded image.
        
        Args:
            uploaded_file: Uploaded file object (e.g., from Streamlit file_uploader)
            original_filename: Original filename
            
        Returns:
            Dictionary with validation and save results
        """
        try:
            self._init_components()
            
            # Generate unique filename
            unique_filename = self.generate_unique_filename(original_filename)
            image_path = os.path.join(USER_IMAGES_PATH, unique_filename)
            
            # Check file size
            file_size = len(uploaded_file.getvalue()) if hasattr(uploaded_file, 'getvalue') else 0
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                return {
                    "success": False,
                    "error": f"File size ({file_size / (1024*1024):.1f}MB) exceeds {MAX_FILE_SIZE_MB}MB limit"
                }
            
            # Save uploaded file
            with open(image_path, 'wb') as f:
                if hasattr(uploaded_file, 'getvalue'):
                    f.write(uploaded_file.getvalue())
                else:
                    shutil.copyfileobj(uploaded_file, f)
            
            # Validate saved image
            validation = self.feature_extractor.validate_image(image_path)
            if not validation["valid"]:
                # Remove invalid file
                if os.path.exists(image_path):
                    os.remove(image_path)
                return {
                    "success": False,
                    "error": validation["error"]
                }
            
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size
            
            return {
                "success": True,
                "image_path": image_path,
                "unique_filename": unique_filename,
                "original_filename": original_filename,
                "file_size_bytes": file_size,
                "image_width": width,
                "image_height": height
            }
            
        except Exception as e:
            # Clean up any partially saved file
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
            
            return {
                "success": False,
                "error": f"Image processing failed: {str(e)}"
            }
    
    def process_user_image(self, image_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from user image and store metadata.
        
        Args:
            image_info: Dictionary with image information from validate_and_save_image
            
        Returns:
            Dictionary with processing results including image_id
        """
        try:
            self._init_components()
            
            image_path = image_info["image_path"]
            
            # Extract features
            features = self.feature_extractor.extract_features(image_path)
            
            # Save embedding
            embedding_filename = os.path.splitext(image_info["unique_filename"])[0] + ".npy"
            embedding_path = os.path.join(USER_EMBEDDINGS_PATH, embedding_filename)
            np.save(embedding_path, features)
            
            # Store in database
            image_id = self.database_manager.store_user_image(
                filename=image_info["unique_filename"],
                original_name=image_info["original_filename"],
                embedding_path=embedding_path,
                file_size_bytes=image_info["file_size_bytes"],
                image_width=image_info["image_width"],
                image_height=image_info["image_height"]
            )
            
            return {
                "success": True,
                "image_id": image_id,
                "embedding_path": embedding_path,
                "features_shape": features.shape,
                "features_stats": {
                    "min": float(features.min()),
                    "max": float(features.max()),
                    "mean": float(features.mean())
                }
            }
            
        except Exception as e:
            # Clean up embedding file if created
            if 'embedding_path' in locals() and os.path.exists(embedding_path):
                os.remove(embedding_path)
            
            return {
                "success": False,
                "error": f"Feature extraction failed: {str(e)}"
            }
    
    def get_recommendations(self, user_image_ids: List[int], num_recommendations: int = 5) -> Dict[str, Any]:
        """
        Get food recommendations based on user images.
        
        Args:
            user_image_ids: List of user image IDs to base recommendations on
            num_recommendations: Number of recommendations to return
            
        Returns:
            Dictionary with recommendations and metadata
        """
        try:
            self._init_components()
            
            if not user_image_ids:
                return {
                    "success": False,
                    "error": "No user images provided"
                }
            
            # Check if FAISS index is loaded
            if self.similarity_engine.index is None:
                return {
                    "success": False,
                    "error": "Food-101 index not available. Run preprocessing first."
                }
            
            # Load user embeddings
            user_embeddings = []
            valid_image_ids = []
            
            for image_id in user_image_ids:
                try:
                    # Get embedding path from database
                    user_images = self.database_manager.get_user_images(processed_only=True)
                    user_image = next((img for img in user_images if img['id'] == image_id), None)
                    
                    if user_image and user_image['embedding_path']:
                        embedding = np.load(user_image['embedding_path'])
                        user_embeddings.append(embedding)
                        valid_image_ids.append(image_id)
                except Exception as e:
                    logging.warning(f"Failed to load embedding for image {image_id}: {str(e)}")
                    continue
            
            if not user_embeddings:
                return {
                    "success": False,
                    "error": "No valid user embeddings found"
                }
            
            # Get recommendations using similarity engine
            recommendations = self.similarity_engine.search_by_multiple_embeddings(
                user_embeddings, k=num_recommendations
            )
            
            return {
                "success": True,
                "recommendations": recommendations,
                "num_user_images": len(valid_image_ids),
                "user_image_ids": valid_image_ids,
                "aggregation_method": "simple_average"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Recommendation generation failed: {str(e)}"
            }
    
    def store_feedback(self, user_image_id: int, food_class: str, food_name: str, 
                      feedback: int, similarity_score: float) -> Dict[str, Any]:
        """
        Store user feedback for a recommendation.
        
        Args:
            user_image_id: ID of user image that generated the recommendation
            food_class: Food class (snake_case)
            food_name: Display name of the food
            feedback: 1 for like, -1 for dislike, 0 for neutral
            similarity_score: Original similarity score
            
        Returns:
            Dictionary with feedback storage result
        """
        try:
            self._init_components()
            
            self.database_manager.store_feedback(
                user_image_id=user_image_id,
                food_class=food_class,
                food_name=food_name,
                feedback=feedback,
                similarity_score=similarity_score
            )
            
            return {
                "success": True,
                "message": f"Feedback stored: {food_name} -> {feedback}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to store feedback: {str(e)}"
            }
    
    def delete_user_image(self, image_id: int) -> Dict[str, Any]:
        """
        Delete a user image including files and database records.
        
        Args:
            image_id: Database ID of the image to delete
            
        Returns:
            Dictionary with deletion result
        """
        try:
            self._init_components()
            
            # Get image details before deletion
            details_result = self.database_manager.get_user_image_details(image_id)
            if not details_result["success"]:
                return details_result
            
            image_details = details_result["image_details"]
            
            # Delete from database first
            db_result = self.database_manager.delete_user_image(image_id)
            if not db_result["success"]:
                return db_result
            
            deleted_files = []
            failed_deletions = []
            
            # Clean up image file
            image_path = os.path.join(USER_IMAGES_PATH, image_details["filename"])
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    deleted_files.append(image_path)
                except Exception as e:
                    failed_deletions.append(f"Image file: {str(e)}")
            
            # Clean up embedding file
            if image_details["embedding_path"] and os.path.exists(image_details["embedding_path"]):
                try:
                    os.remove(image_details["embedding_path"])
                    deleted_files.append(image_details["embedding_path"])
                except Exception as e:
                    failed_deletions.append(f"Embedding file: {str(e)}")
            
            result = {
                "success": True,
                "message": db_result["message"],
                "deleted_files": deleted_files,
                "failed_file_deletions": failed_deletions,
                "feedback_deleted": db_result["feedback_deleted"]
            }
            
            if failed_deletions:
                result["warning"] = f"Some files could not be deleted: {'; '.join(failed_deletions)}"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete user image: {str(e)}"
            }
    
    def get_user_image_details(self, image_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a user image.
        
        Args:
            image_id: Database ID of the image
            
        Returns:
            Dictionary with image details and metadata
        """
        try:
            self._init_components()
            
            result = self.database_manager.get_user_image_details(image_id)
            if not result["success"]:
                return result
            
            image_details = result["image_details"]
            
            # Add file existence check
            image_path = os.path.join(USER_IMAGES_PATH, image_details["filename"])
            image_details["file_exists"] = os.path.exists(image_path)
            image_details["full_image_path"] = image_path
            
            if image_details["embedding_path"]:
                image_details["embedding_exists"] = os.path.exists(image_details["embedding_path"])
            else:
                image_details["embedding_exists"] = False
            
            return {
                "success": True,
                "image_details": image_details,
                "feedback_history": result["feedback_history"],
                "feedback_count": result["feedback_count"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get image details: {str(e)}"
            }

    def get_user_images_summary(self) -> Dict[str, Any]:
        """
        Get summary of user images and processing status.
        
        Returns:
            Dictionary with user images summary
        """
        try:
            self._init_components()
            
            user_images = self.database_manager.get_user_images()
            db_stats = self.database_manager.get_database_stats()
            feedback_stats = self.database_manager.get_feedback_stats()
            
            return {
                "success": True,
                "total_images": len(user_images),
                "processed_images": db_stats["num_processed_images"],
                "total_feedback": feedback_stats["total_feedback"],
                "likes": feedback_stats["likes"],
                "dislikes": feedback_stats["dislikes"],
                "recent_images": user_images[:5]  # Most recent 5 images
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get user images summary: {str(e)}"
            }
    
    def process_uploaded_image_complete(self, uploaded_file, original_filename: str, 
                                      get_recommendations: bool = True, 
                                      num_recommendations: int = 5) -> Dict[str, Any]:
        """
        Complete pipeline: validate, save, process, and get recommendations.
        
        Args:
            uploaded_file: Uploaded file object
            original_filename: Original filename
            get_recommendations: Whether to generate recommendations
            num_recommendations: Number of recommendations to generate
            
        Returns:
            Dictionary with complete processing results
        """
        try:
            # Step 1: Validate and save image
            save_result = self.validate_and_save_image(uploaded_file, original_filename)
            if not save_result["success"]:
                return save_result
            
            # Step 2: Process image (extract features)
            process_result = self.process_user_image(save_result)
            if not process_result["success"]:
                # Clean up saved image
                if os.path.exists(save_result["image_path"]):
                    os.remove(save_result["image_path"])
                return process_result
            
            result = {
                "success": True,
                "image_id": process_result["image_id"],
                "image_path": save_result["image_path"],
                "unique_filename": save_result["unique_filename"],
                "embedding_path": process_result["embedding_path"],
                "features_stats": process_result["features_stats"]
            }
            
            # Step 3: Get recommendations if requested
            if get_recommendations:
                rec_result = self.get_recommendations([process_result["image_id"]], num_recommendations)
                result["recommendations"] = rec_result
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Complete processing failed: {str(e)}"
            }


def test_user_image_processor():
    """Test function for UserImageProcessor."""
    print("Testing UserImageProcessor...")
    
    try:
        processor = UserImageProcessor()
        
        # Test with a sample Food-101 image (simulating user upload)
        test_image_path = "food-101/images/apple_pie/1005649.jpg"
        if not os.path.exists(test_image_path):
            print("✗ Test image not found, skipping test")
            return
        
        # Simulate file upload by reading the image
        with open(test_image_path, 'rb') as f:
            test_data = f.read()
        
        class MockUploadedFile:
            def __init__(self, data):
                self.data = data
            def getvalue(self):
                return self.data
        
        mock_file = MockUploadedFile(test_data)
        
        # Test complete processing pipeline
        result = processor.process_uploaded_image_complete(
            mock_file, 
            "test_apple_pie.jpg",
            get_recommendations=True,
            num_recommendations=5
        )
        
        if result["success"]:
            print(f"✓ Image processed successfully:")
            print(f"  Image ID: {result['image_id']}")
            print(f"  Features shape: {result['features_stats']}")
            
            if "recommendations" in result:
                rec_result = result["recommendations"]
                if rec_result["success"]:
                    recommendations = rec_result["recommendations"]
                    print(f"  Recommendations: {len(recommendations)} foods found")
                    for i, rec in enumerate(recommendations[:3]):
                        print(f"    {i+1}. {rec['food_name']} (similarity: {rec['similarity_score']:.3f})")
                else:
                    print(f"  ✗ Recommendations failed: {rec_result['error']}")
        else:
            print(f"✗ Processing failed: {result['error']}")
        
        # Test user images summary
        summary = processor.get_user_images_summary()
        if summary["success"]:
            print(f"✓ User images summary: {summary['total_images']} total, {summary['processed_images']} processed")
        
        print("UserImageProcessor test completed!")
        
    except Exception as e:
        print(f"✗ UserImageProcessor test failed: {str(e)}")


if __name__ == "__main__":
    test_user_image_processor()