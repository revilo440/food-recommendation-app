"""
Food Recommendation App - Streamlit Interface
A minimalistic, local-first food recommendation system
"""
import streamlit as st
import os
from PIL import Image
from utils.user_image_processor import UserImageProcessor
from data.database_manager import DatabaseManager
from config import USER_IMAGES_PATH
import json

# Fix OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Page configuration
st.set_page_config(
    page_title="Food Recommendations", 
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalistic design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .recommendation-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
    }
    
    .food-image {
        border-radius: 8px;
        width: 100%;
        height: 200px;
        object-fit: cover;
    }
    
    .similarity-score {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .feedback-buttons {
        margin-top: 0.5rem;
    }
    
    .stats-container {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .user-image-gallery {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .user-image-thumb {
        width: 80px;
        height: 80px;
        border-radius: 4px;
        object-fit: cover;
        border: 2px solid #e0e0e0;
    }
    
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #c62828;
    }
    
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'processor' not in st.session_state:
        st.session_state.processor = UserImageProcessor()
    
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    
    if 'current_recommendations' not in st.session_state:
        st.session_state.current_recommendations = None
    
    if 'show_feedback_success' not in st.session_state:
        st.session_state.show_feedback_success = None
    
    if 'selected_image_for_viewing' not in st.session_state:
        st.session_state.selected_image_for_viewing = None
    
    if 'show_image_manager' not in st.session_state:
        st.session_state.show_image_manager = False
    
    if 'deletion_success' not in st.session_state:
        st.session_state.deletion_success = None


def display_header():
    """Display the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>Food Recommendation App</h1>
        <p style="color: #666; font-size: 1.1rem;">
            Upload your food photos and discover similar dishes from our curated collection
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_upload_section():
    """Display the image upload section."""
    st.markdown("### 📸 Upload Your Food Image")
    
    uploaded_file = st.file_uploader(
        "Choose a food image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of your food. Supported formats: JPG, PNG (max 15MB)",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Display uploaded image preview
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Get Recommendations", use_container_width=True, type="primary"):
                process_uploaded_image(uploaded_file)


def process_uploaded_image(uploaded_file):
    """Process the uploaded image and get recommendations."""
    with st.spinner("Processing your image and finding recommendations..."):
        try:
            # Process image through our ML pipeline
            result = st.session_state.processor.process_uploaded_image_complete(
                uploaded_file,
                uploaded_file.name,
                get_recommendations=True,
                num_recommendations=st.session_state.get('num_recommendations', 5)
            )
            
            if result["success"]:
                # Store processed image info
                st.session_state.processed_images.append({
                    'id': result['image_id'],
                    'filename': result['unique_filename'],
                    'path': result['image_path']
                })
                
                # Store recommendations
                if "recommendations" in result and result["recommendations"]["success"]:
                    st.session_state.current_recommendations = {
                        'recommendations': result["recommendations"]["recommendations"],
                        'user_image_id': result['image_id']
                    }
                    
                    st.success("Image processed successfully! Check out your recommendations below.")
                else:
                    st.error("Failed to generate recommendations. Please try again.")
            else:
                st.error(f"Processing failed: {result['error']}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


def display_settings():
    """Display settings panel."""
    with st.expander("Settings", expanded=False):
        num_recs = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=st.session_state.get('num_recommendations', 5),
            help="How many food recommendations to show"
        )
        st.session_state.num_recommendations = num_recs
        
        # Regenerate recommendations if images exist and setting changed
        if (st.session_state.processed_images and 
            st.session_state.get('last_num_recs') != num_recs):
            
            if st.button("🔄 Update Recommendations"):
                regenerate_recommendations()
                st.session_state.last_num_recs = num_recs


def regenerate_recommendations():
    """Regenerate recommendations with current settings."""
    if not st.session_state.processed_images:
        return
        
    with st.spinner("Updating recommendations..."):
        try:
            image_ids = [img['id'] for img in st.session_state.processed_images]
            rec_result = st.session_state.processor.get_recommendations(
                image_ids, 
                num_recommendations=st.session_state.num_recommendations
            )
            
            if rec_result["success"]:
                st.session_state.current_recommendations = {
                    'recommendations': rec_result["recommendations"],
                    'user_image_id': image_ids[0] if image_ids else None
                }
                st.success("Recommendations updated!")
            else:
                st.error(f"Failed to update recommendations: {rec_result['error']}")
                
        except Exception as e:
            st.error(f"Error updating recommendations: {str(e)}")


def display_user_images():
    """Display gallery of user uploaded images."""
    if not st.session_state.processed_images:
        return
    
    # Header with management button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📷 Your Uploaded Images")
    with col2:
        if st.button("Manage Images", use_container_width=True):
            st.session_state.show_image_manager = True
            st.rerun()
    
    # Show thumbnail gallery
    cols = st.columns(min(len(st.session_state.processed_images), 5))
    
    for i, img_info in enumerate(st.session_state.processed_images[-5:]):  # Show last 5 images
        with cols[i % 5]:
            try:
                if os.path.exists(img_info['path']):
                    image = Image.open(img_info['path'])
                    
                    # Clickable image for viewing
                    if st.button(f"👁️", key=f"view_thumb_{img_info['id']}", help="Click to view full image"):
                        st.session_state.selected_image_for_viewing = img_info['id']
                        st.rerun()
                    
                    st.image(image, caption=f"Image {img_info['id']}", use_container_width=True)
                else:
                    st.caption(f"Image {img_info['id']} (file not found)")
            except Exception:
                st.caption(f"Image {img_info['id']} (error loading)")


def handle_feedback(food_class, food_name, similarity_score, user_image_id, feedback_type):
    """Handle user feedback on recommendations."""
    feedback_value = 1 if feedback_type == "like" else -1
    
    try:
        result = st.session_state.processor.store_feedback(
            user_image_id=user_image_id,
            food_class=food_class,
            food_name=food_name,
            feedback=feedback_value,
            similarity_score=similarity_score
        )
        
        if result["success"]:
            emoji = "👍" if feedback_type == "like" else "👎"
            st.session_state.show_feedback_success = f"{emoji} Thanks for your feedback on {food_name}!"
            st.rerun()
        else:
            st.error(f"Failed to save feedback: {result['error']}")
            
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")


def display_recommendations():
    """Display food recommendations with images and feedback buttons."""
    if not st.session_state.current_recommendations:
        return
    
    recommendations = st.session_state.current_recommendations['recommendations']
    user_image_id = st.session_state.current_recommendations['user_image_id']
    
    if not recommendations:
        st.info("No recommendations found. Try uploading a different image.")
        return
    
    st.markdown("### 🍽️ Recommended Foods")
    
    # Show feedback success message
    if st.session_state.show_feedback_success:
        st.success(st.session_state.show_feedback_success)
        st.session_state.show_feedback_success = None
    
    # Display recommendations in grid
    for i in range(0, len(recommendations), 2):
        cols = st.columns(2)
        
        for j, col in enumerate(cols):
            if i + j < len(recommendations):
                rec = recommendations[i + j]
                
                with col:
                    # Create recommendation card
                    with st.container():
                        # Try to display Food-101 image
                        image_path = rec.get('image_path', '')
                        if image_path and os.path.exists(image_path):
                            try:
                                food_image = Image.open(image_path)
                                st.image(food_image, use_container_width=True)
                            except Exception:
                                st.info("Image not available")
                        else:
                            st.info("Image not available")
                        
                        # Food info
                        st.markdown(f"**{rec['food_name']}**")
                        
                        # Similarity score
                        similarity_pct = rec['similarity_score'] * 100
                        st.markdown(f"🎯 **{similarity_pct:.1f}%** match")
                        
                        # Feedback buttons
                        col_like, col_dislike = st.columns(2)
                        
                        with col_like:
                            if st.button(
                                "👍 Like", 
                                key=f"like_{rec['food_class']}_{i+j}",
                                use_container_width=True
                            ):
                                handle_feedback(
                                    rec['food_class'], 
                                    rec['food_name'], 
                                    rec['similarity_score'],
                                    user_image_id, 
                                    "like"
                                )
                        
                        with col_dislike:
                            if st.button(
                                "👎 Dislike", 
                                key=f"dislike_{rec['food_class']}_{i+j}",
                                use_container_width=True
                            ):
                                handle_feedback(
                                    rec['food_class'], 
                                    rec['food_name'], 
                                    rec['similarity_score'],
                                    user_image_id, 
                                    "dislike"
                                )
                        
                        st.markdown("---")


def display_stats():
    """Display app statistics."""
    try:
        summary = st.session_state.processor.get_user_images_summary()
        
        if summary["success"]:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📸 Images Uploaded", summary['total_images'])
            
            with col2:
                st.metric("🔄 Images Processed", summary['processed_images'])
            
            with col3:
                st.metric("👍 Likes Given", summary['likes'])
            
            with col4:
                st.metric("👎 Dislikes Given", summary['dislikes'])
    
    except Exception as e:
        st.error(f"Error loading stats: {str(e)}")


def display_image_viewer():
    """Display full-size image viewer modal."""
    if st.session_state.selected_image_for_viewing is None:
        return
    
    image_id = st.session_state.selected_image_for_viewing
    
    # Get image details
    details_result = st.session_state.processor.get_user_image_details(image_id)
    
    if not details_result["success"]:
        st.error(f"Error loading image: {details_result['error']}")
        st.session_state.selected_image_for_viewing = None
        return
    
    image_details = details_result["image_details"]
    feedback_history = details_result["feedback_history"]
    
    # Modal-style container
    with st.container():
        st.markdown("---")
        st.markdown(f"### 🖼️ Image Viewer - {image_details['original_name']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display the image
            if image_details["file_exists"]:
                try:
                    image = Image.open(image_details["full_image_path"])
                    st.image(image, caption=f"Uploaded: {image_details['upload_timestamp']}", use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
            else:
                st.error("Image file not found")
        
        with col2:
            # Image metadata
            st.markdown("**Image Details**")
            st.write(f"**Original Name:** {image_details['original_name']}")
            st.write(f"**Upload Date:** {image_details['upload_timestamp']}")
            st.write(f"**Size:** {image_details['image_width']}×{image_details['image_height']}px")
            
            if image_details['file_size_bytes']:
                size_mb = image_details['file_size_bytes'] / (1024 * 1024)
                st.write(f"**File Size:** {size_mb:.2f} MB")
            
            st.write(f"**Processed:** {'✅' if image_details['processed'] else '❌'}")
            st.write(f"**Embedding:** {'✅' if image_details['embedding_exists'] else '❌'}")
            
            # Feedback history
            if feedback_history:
                st.markdown("**Feedback History**")
                for feedback in feedback_history[:5]:  # Show last 5 feedback items
                    emoji = "👍" if feedback['feedback'] == 1 else "👎" if feedback['feedback'] == -1 else "😐"
                    st.write(f"{emoji} {feedback['food_name']} ({feedback['similarity_score']:.3f})")
            
            # Action buttons
            st.markdown("**Actions**")
            
            col_regen, col_delete = st.columns(2)
            
            with col_regen:
                if st.button("🔄 Regenerate Recs", key=f"regen_{image_id}", use_container_width=True):
                    regenerate_single_image_recommendations(image_id)
            
            with col_delete:
                if st.button("🗑️ Delete Image", key=f"delete_{image_id}", use_container_width=True, type="secondary"):
                    st.session_state.confirm_delete_id = image_id
                    st.rerun()
        
        # Close button
        if st.button("Close Viewer", use_container_width=True):
            st.session_state.selected_image_for_viewing = None
            st.rerun()
        
        st.markdown("---")


def display_image_manager():
    """Display the image management interface."""
    if not st.session_state.show_image_manager:
        return
    
    st.markdown("## 🗂️ Image Manager")
    
    # Get all user images
    try:
        summary = st.session_state.processor.get_user_images_summary()
        if not summary["success"]:
            st.error(f"Error loading images: {summary['error']}")
            return
        
        # Get detailed list
        user_images = st.session_state.processor.database_manager.get_user_images()
        
        if not user_images:
            st.info("📷 No images uploaded yet")
            if st.button("Close Manager"):
                st.session_state.show_image_manager = False
                st.rerun()
            return
        
        # Show deletion success message
        if st.session_state.deletion_success:
            st.success(st.session_state.deletion_success)
            st.session_state.deletion_success = None
        
        # Display images in a table-like format
        for img in user_images:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 1, 1])
                
                # Thumbnail
                with col1:
                    img_path = os.path.join(USER_IMAGES_PATH, img['filename'])
                    if os.path.exists(img_path):
                        try:
                            thumbnail = Image.open(img_path)
                            st.image(thumbnail, width=80)
                        except:
                            st.write("🖼️")
                    else:
                        st.write("❌")
                
                # Image info
                with col2:
                    st.write(f"**{img['original_name']}**")
                    st.caption(f"ID: {img['id']} | {img['upload_timestamp']}")
                    if img['image_width'] and img['image_height']:
                        st.caption(f"{img['image_width']}×{img['image_height']}px")
                
                # Status
                with col3:
                    status_text = "✅ Processed" if img['processed'] else "⏳ Processing"
                    st.write(status_text)
                    
                    # Check if files exist
                    img_exists = os.path.exists(os.path.join(USER_IMAGES_PATH, img['filename']))
                    emb_exists = img['embedding_path'] and os.path.exists(img['embedding_path'])
                    st.caption(f"File: {'✅' if img_exists else '❌'} | Embedding: {'✅' if emb_exists else '❌'}")
                
                # View button
                with col4:
                    if st.button("👁️ View", key=f"view_{img['id']}", use_container_width=True):
                        st.session_state.selected_image_for_viewing = img['id']
                        st.session_state.show_image_manager = False
                        st.rerun()
                
                # Delete button
                with col5:
                    if st.button("🗑️", key=f"del_{img['id']}", use_container_width=True, help="Delete image"):
                        st.session_state.confirm_delete_id = img['id']
                        st.rerun()
                
                st.markdown("---")
        
        # Close manager
        if st.button("Close Manager", use_container_width=True):
            st.session_state.show_image_manager = False
            st.rerun()
            
    except Exception as e:
        st.error(f"Error in image manager: {str(e)}")


def handle_image_deletion():
    """Handle image deletion confirmation."""
    if 'confirm_delete_id' not in st.session_state:
        return
    
    image_id = st.session_state.confirm_delete_id
    
    # Get image details for confirmation
    details_result = st.session_state.processor.get_user_image_details(image_id)
    
    if details_result["success"]:
        image_name = details_result["image_details"]["original_name"]
        feedback_count = details_result["feedback_count"]
        
        # Confirmation dialog
        st.warning(f"⚠️ **Delete Confirmation**")
        st.write(f"Are you sure you want to delete **{image_name}**?")
        
        if feedback_count > 0:
            st.write(f"This will also delete {feedback_count} associated feedback records.")
        
        st.write("This action cannot be undone.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🗑️ Yes, Delete", type="primary", use_container_width=True):
                # Perform deletion
                delete_result = st.session_state.processor.delete_user_image(image_id)
                
                if delete_result["success"]:
                    st.session_state.deletion_success = f"✅ {delete_result['message']}"
                    
                    # Update processed images list
                    st.session_state.processed_images = [
                        img for img in st.session_state.processed_images 
                        if img['id'] != image_id
                    ]
                    
                    # Clear current recommendations if they were from this image
                    if (st.session_state.current_recommendations and 
                        st.session_state.current_recommendations.get('user_image_id') == image_id):
                        st.session_state.current_recommendations = None
                    
                else:
                    st.error(f"Deletion failed: {delete_result['error']}")
                
                # Clear confirmation
                del st.session_state.confirm_delete_id
                st.rerun()
        
        with col2:
            if st.button("Cancel", use_container_width=True):
                del st.session_state.confirm_delete_id
                st.rerun()
    
    else:
        st.error(f"Error getting image details: {details_result['error']}")
        del st.session_state.confirm_delete_id


def regenerate_single_image_recommendations(image_id):
    """Regenerate recommendations for a single image."""
    with st.spinner("Regenerating recommendations..."):
        try:
            rec_result = st.session_state.processor.get_recommendations(
                [image_id], 
                num_recommendations=st.session_state.get('num_recommendations', 5)
            )
            
            if rec_result["success"]:
                st.session_state.current_recommendations = {
                    'recommendations': rec_result["recommendations"],
                    'user_image_id': image_id
                }
                st.success("✅ Recommendations updated!")
                st.session_state.selected_image_for_viewing = None
                st.rerun()
            else:
                st.error(f"Failed to regenerate recommendations: {rec_result['error']}")
                
        except Exception as e:
            st.error(f"Error regenerating recommendations: {str(e)}")


def check_system_status():
    """Check if the system is properly initialized."""
    try:
        # Initialize processor
        processor = UserImageProcessor()
        processor._init_components()
        
        # Check FAISS index
        index_info = processor.similarity_engine.get_index_info()
        
        if not index_info["index_loaded"]:
            st.error("""
            ❌ **System Not Ready**
            
            The Food-101 index is not loaded. Please run the preprocessing pipeline first:
            
            ```bash
            python -m data.food101_processor --full-setup --max-images-per-class 10
            ```
            """)
            st.stop()
        
        return True
        
    except Exception as e:
        st.error(f"""
        ❌ **System Error**
        
        Failed to initialize the recommendation system: {str(e)}
        
        Please ensure all dependencies are installed and the system is properly set up.
        """)
        st.stop()


def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Check system status
    check_system_status()
    
    # Handle image deletion confirmation (must be before other UI components)
    handle_image_deletion()
    
    # Check if we should show image manager or image viewer
    if st.session_state.show_image_manager:
        display_image_manager()
        return
    
    if st.session_state.selected_image_for_viewing:
        display_image_viewer()
        return
    
    # Display header
    display_header()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload section
        display_upload_section()
        
        # User images gallery
        display_user_images()
        
        # Recommendations
        display_recommendations()
    
    with col2:
        # Settings
        display_settings()
        
        # Statistics
        st.markdown("### 📊 Statistics")
        display_stats()
        
        # System info
        with st.expander("ℹ️ System Info", expanded=False):
            try:
                processor = st.session_state.processor
                processor._init_components()
                index_info = processor.similarity_engine.get_index_info()
                
                st.write(f"**FAISS Index**: {index_info['num_embeddings']} embeddings")
                st.write(f"**Food Categories**: {index_info['num_food_classes']} classes")
                st.write(f"**Model**: ResNet50 (2048-dim features)")
                st.write(f"**Database**: SQLite (local)")
                
            except Exception as e:
                st.write(f"Error loading system info: {str(e)}")


if __name__ == "__main__":
    main()