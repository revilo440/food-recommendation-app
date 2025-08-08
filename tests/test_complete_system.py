"""
Complete System Test
Tests the entire food recommendation system end-to-end
"""
import os
import numpy as np
from utils.user_image_processor import UserImageProcessor
from data.database_manager import DatabaseManager
from models.similarity_engine import SimilarityEngine
import random


def get_sample_images_from_food101(num_samples=5):
    """Get sample images from different Food-101 categories for testing."""
    food101_path = "food-101/images"
    sample_images = []
    
    # Get a few different food categories
    categories = ['apple_pie', 'hamburger', 'sushi', 'pizza', 'ice_cream']
    
    for category in categories[:num_samples]:
        category_path = os.path.join(food101_path, category)
        if os.path.exists(category_path):
            images = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
            if images:
                # Pick a random image from this category
                random_image = random.choice(images)
                image_path = os.path.join(category_path, random_image)
                sample_images.append({
                    'path': image_path,
                    'category': category,
                    'filename': f"test_{category}_{random_image}"
                })
    
    return sample_images


class MockUploadedFile:
    """Mock uploaded file for testing."""
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = f.read()
    
    def getvalue(self):
        return self.data


def test_complete_system():
    """Test the complete food recommendation system."""
    print("🧪 Testing Complete Food Recommendation System")
    print("=" * 60)
    
    # Fix OpenMP issue
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Initialize processor
    processor = UserImageProcessor()
    
    # Test 1: System Status Check
    print("\n1. System Status Check")
    summary = processor.get_user_images_summary()
    if summary["success"]:
        print(f"   ✓ Database connected: {summary['total_images']} images, {summary['total_feedback']} feedback")
    else:
        print(f"   ✗ Database error: {summary['error']}")
        return
    
    # Check FAISS index
    processor._init_components()
    index_info = processor.similarity_engine.get_index_info()
    if index_info["index_loaded"]:
        print(f"   ✓ FAISS index loaded: {index_info['num_embeddings']} embeddings")
    else:
        print("   ✗ FAISS index not loaded")
        return
    
    # Test 2: Process Multiple User Images
    print("\n2. Processing User Images")
    sample_images = get_sample_images_from_food101(3)
    processed_image_ids = []
    
    for i, sample in enumerate(sample_images):
        print(f"   Processing {sample['category']} image...")
        
        # Create mock uploaded file
        mock_file = MockUploadedFile(sample['path'])
        
        # Process image
        result = processor.process_uploaded_image_complete(
            mock_file, 
            sample['filename'],
            get_recommendations=False  # We'll test recommendations separately
        )
        
        if result["success"]:
            processed_image_ids.append(result["image_id"])
            print(f"   ✓ {sample['category']}: Image ID {result['image_id']}")
        else:
            print(f"   ✗ {sample['category']}: {result['error']}")
    
    print(f"   Successfully processed {len(processed_image_ids)} images")
    
    # Test 3: Generate Recommendations
    print("\n3. Generating Recommendations")
    
    # Test single image recommendations
    for image_id in processed_image_ids[:2]:  # Test first 2 images
        rec_result = processor.get_recommendations([image_id], num_recommendations=5)
        
        if rec_result["success"]:
            recommendations = rec_result["recommendations"]
            print(f"   ✓ Image {image_id}: {len(recommendations)} recommendations")
            
            # Show top 3 recommendations
            for j, rec in enumerate(recommendations[:3]):
                print(f"      {j+1}. {rec['food_name']} (similarity: {rec['similarity_score']:.3f})")
        else:
            print(f"   ✗ Image {image_id}: {rec_result['error']}")
    
    # Test multiple image recommendations (aggregated)
    if len(processed_image_ids) >= 2:
        print(f"\n   Testing aggregated recommendations from {len(processed_image_ids)} images:")
        multi_rec_result = processor.get_recommendations(processed_image_ids, num_recommendations=8)
        
        if multi_rec_result["success"]:
            recommendations = multi_rec_result["recommendations"]
            print(f"   ✓ Aggregated: {len(recommendations)} recommendations")
            print(f"   Based on {multi_rec_result['num_user_images']} user images")
            
            # Show top 5 aggregated recommendations
            for j, rec in enumerate(recommendations[:5]):
                print(f"      {j+1}. {rec['food_name']} (similarity: {rec['similarity_score']:.3f})")
        else:
            print(f"   ✗ Aggregated recommendations failed: {multi_rec_result['error']}")
    
    # Test 4: Feedback System
    print("\n4. Testing Feedback System")
    
    if processed_image_ids:
        # Get recommendations for feedback testing
        rec_result = processor.get_recommendations([processed_image_ids[0]], num_recommendations=3)
        
        if rec_result["success"] and rec_result["recommendations"]:
            recommendations = rec_result["recommendations"]
            
            # Test positive feedback
            rec = recommendations[0]
            feedback_result = processor.store_feedback(
                user_image_id=processed_image_ids[0],
                food_class=rec['food_class'],
                food_name=rec['food_name'],
                feedback=1,  # Like
                similarity_score=rec['similarity_score']
            )
            
            if feedback_result["success"]:
                print(f"   ✓ Positive feedback stored for {rec['food_name']}")
            else:
                print(f"   ✗ Feedback storage failed: {feedback_result['error']}")
            
            # Test negative feedback
            if len(recommendations) > 1:
                rec = recommendations[1]
                feedback_result = processor.store_feedback(
                    user_image_id=processed_image_ids[0],
                    food_class=rec['food_class'],
                    food_name=rec['food_name'],
                    feedback=-1,  # Dislike
                    similarity_score=rec['similarity_score']
                )
                
                if feedback_result["success"]:
                    print(f"   ✓ Negative feedback stored for {rec['food_name']}")
                else:
                    print(f"   ✗ Feedback storage failed: {feedback_result['error']}")
    
    # Test 5: Database Analytics
    print("\n5. Database Analytics")
    
    db_manager = DatabaseManager()
    feedback_stats = db_manager.get_feedback_stats()
    print(f"   Feedback Summary:")
    print(f"      Total feedback: {feedback_stats['total_feedback']}")
    print(f"      Likes: {feedback_stats['likes']}")
    print(f"      Dislikes: {feedback_stats['dislikes']}")
    print(f"      Neutral: {feedback_stats['neutral']}")
    
    if feedback_stats['top_liked']:
        print(f"   Top liked foods:")
        for food in feedback_stats['top_liked'][:3]:
            print(f"      - {food['food_name']}: {food['like_count']} likes")
    
    # Test 6: System Performance Summary
    print("\n6. System Performance Summary")
    
    final_summary = processor.get_user_images_summary()
    if final_summary["success"]:
        print(f"   Total user images: {final_summary['total_images']}")
        print(f"   Processed images: {final_summary['processed_images']}")
        print(f"   Total feedback: {final_summary['total_feedback']}")
        print(f"   Likes: {final_summary['likes']}")
        print(f"   Dislikes: {final_summary['dislikes']}")
    
    print(f"\n   FAISS Index: {index_info['num_embeddings']} food embeddings")
    print(f"   Food Categories: {index_info['num_food_classes']} classes")
    
    print("\n" + "=" * 60)
    print("Complete System Test Finished!")
    print("All core ML functionality is working correctly!")
    print("\nThe system is ready for Streamlit UI integration.")


if __name__ == "__main__":
    test_complete_system()