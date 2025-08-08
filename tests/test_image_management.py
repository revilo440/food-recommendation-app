"""
Test Image Management Functionality
"""
import os
from utils.user_image_processor import UserImageProcessor
from data.database_manager import DatabaseManager

# Fix OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_image_management():
    """Test the image management features."""
    print("🧪 Testing Image Management Functionality")
    print("=" * 50)
    
    processor = UserImageProcessor()
    db_manager = DatabaseManager()
    
    # Test 1: Get current user images
    print("\n1. Current User Images")
    summary = processor.get_user_images_summary()
    if summary["success"]:
        print(f"   Total images: {summary['total_images']}")
        print(f"   Processed: {summary['processed_images']}")
    
    user_images = db_manager.get_user_images()
    print(f"   Found {len(user_images)} images in database")
    
    if not user_images:
        print("   ℹ️  No images to test with. Upload some images first.")
        return
    
    # Test 2: Get detailed info for first image
    test_image = user_images[0]
    image_id = test_image['id']
    
    print(f"\n2. Testing Image Details (ID: {image_id})")
    details_result = processor.get_user_image_details(image_id)
    
    if details_result["success"]:
        details = details_result["image_details"]
        print(f"   ✓ Original name: {details['original_name']}")
        print(f"   ✓ File exists: {details['file_exists']}")
        print(f"   ✓ Embedding exists: {details['embedding_exists']}")
        print(f"   ✓ Feedback count: {details_result['feedback_count']}")
        
        if details_result['feedback_history']:
            print("   📝 Recent feedback:")
            for feedback in details_result['feedback_history'][:3]:
                emoji = "👍" if feedback['feedback'] == 1 else "👎"
                print(f"      {emoji} {feedback['food_name']} ({feedback['similarity_score']:.3f})")
    else:
        print(f"   Error: {details_result['error']}")
    
    # Test 3: Test deletion (but don't actually delete)
    print(f"\n3. Testing Deletion Logic (Dry Run)")
    print(f"   Would delete image: {test_image['original_name']}")
    print("   Files that would be cleaned up:")
    
    if details_result["success"]:
        details = details_result["image_details"]
        image_path = os.path.join("app_data/user_images", details["filename"])
        print(f"   - Image file: {image_path} ({'exists' if os.path.exists(image_path) else 'missing'})")
        
        if details["embedding_path"]:
            print(f"   - Embedding: {details['embedding_path']} ({'exists' if os.path.exists(details['embedding_path']) else 'missing'})")
        
        print(f"   - Database records: Image + {details_result['feedback_count']} feedback records")
    
    print(f"\n   Deletion test completed (no files were actually deleted)")
    
    # Test 4: Verify all user images have valid paths
    print(f"\n4. File Integrity Check")
    valid_files = 0
    missing_files = 0
    
    for img in user_images:
        img_path = os.path.join("app_data/user_images", img['filename'])
        if os.path.exists(img_path):
            valid_files += 1
        else:
            missing_files += 1
            print(f"   ❌ Missing file: {img['filename']}")
    
    print(f"   ✓ Valid files: {valid_files}")
    print(f"   ❌ Missing files: {missing_files}")
    
    print("\n" + "=" * 50)
    print("✅ Image Management Test Completed!")
    
    if missing_files == 0:
        print("🎉 All files are intact and management functionality is ready!")
    else:
        print("⚠️  Some files are missing - deletion cleanup may be needed.")


if __name__ == "__main__":
    test_image_management()