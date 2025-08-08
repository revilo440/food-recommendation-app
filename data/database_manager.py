"""
Database Manager for SQLite storage of user data and feedback
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

from config import DATABASE_PATH, APP_DATA_PATH


class DatabaseManager:
    """
    Database manager for handling user images, embeddings, and feedback storage.
    Uses SQLite for local-first data storage.
    """
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure app_data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database and create tables
        self._init_database()
        
        logging.info(f"DatabaseManager initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize database and create tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create user_images table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT UNIQUE NOT NULL,
                        original_name TEXT NOT NULL,
                        upload_timestamp DATETIME NOT NULL,
                        embedding_path TEXT,
                        processed BOOLEAN DEFAULT FALSE,
                        file_size_bytes INTEGER,
                        image_width INTEGER,
                        image_height INTEGER
                    )
                ''')
                
                # Create recommendation_feedback table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recommendation_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_image_id INTEGER NOT NULL,
                        food_class TEXT NOT NULL,
                        food_name TEXT NOT NULL,
                        feedback INTEGER NOT NULL, -- 1 for like, -1 for dislike, 0 for neutral
                        similarity_score REAL NOT NULL,
                        feedback_timestamp DATETIME NOT NULL,
                        session_id TEXT,
                        FOREIGN KEY (user_image_id) REFERENCES user_images (id)
                    )
                ''')
                
                # Create user_sessions table for tracking sessions
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        start_timestamp DATETIME NOT NULL,
                        end_timestamp DATETIME,
                        num_uploads INTEGER DEFAULT 0,
                        num_recommendations INTEGER DEFAULT 0
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_images_filename ON user_images(filename)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user_image ON recommendation_feedback(user_image_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_food_class ON recommendation_feedback(food_class)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON user_sessions(session_id)')
                
                conn.commit()
                logging.info("Database tables initialized successfully")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize database: {str(e)}")
    
    def store_user_image(self, filename: str, original_name: str, 
                        embedding_path: str = None, file_size_bytes: int = None,
                        image_width: int = None, image_height: int = None) -> int:
        """
        Store user image metadata in database.
        
        Args:
            filename: Stored filename (unique)
            original_name: Original uploaded filename
            embedding_path: Path to generated embedding file
            file_size_bytes: File size in bytes
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Database ID of the stored image record
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO user_images 
                    (filename, original_name, upload_timestamp, embedding_path, 
                     processed, file_size_bytes, image_width, image_height)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    filename,
                    original_name,
                    datetime.now(),
                    embedding_path,
                    embedding_path is not None,
                    file_size_bytes,
                    image_width,
                    image_height
                ))
                
                image_id = cursor.lastrowid
                conn.commit()
                
                logging.info(f"Stored user image metadata: {filename} (ID: {image_id})")
                return image_id
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(f"Image with filename '{filename}' already exists")
            else:
                raise RuntimeError(f"Database integrity error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to store user image: {str(e)}")
    
    def update_image_embedding(self, image_id: int, embedding_path: str):
        """
        Update the embedding path for a user image.
        
        Args:
            image_id: Database ID of the image
            embedding_path: Path to the generated embedding file
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE user_images 
                    SET embedding_path = ?, processed = TRUE 
                    WHERE id = ?
                ''', (embedding_path, image_id))
                
                if cursor.rowcount == 0:
                    raise ValueError(f"No image found with ID: {image_id}")
                
                conn.commit()
                logging.info(f"Updated embedding path for image ID {image_id}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to update image embedding: {str(e)}")
    
    def store_feedback(self, user_image_id: int, food_class: str, food_name: str,
                      feedback: int, similarity_score: float, session_id: str = None):
        """
        Store user feedback for a recommendation.
        
        Args:
            user_image_id: ID of the user image that generated the recommendation
            food_class: Food class (snake_case)
            food_name: Display name of the food
            feedback: 1 for like, -1 for dislike, 0 for neutral
            similarity_score: Original similarity score
            session_id: Optional session identifier
        """
        try:
            if feedback not in [-1, 0, 1]:
                raise ValueError("Feedback must be -1 (dislike), 0 (neutral), or 1 (like)")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO recommendation_feedback
                    (user_image_id, food_class, food_name, feedback, 
                     similarity_score, feedback_timestamp, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_image_id,
                    food_class,
                    food_name,
                    feedback,
                    similarity_score,
                    datetime.now(),
                    session_id
                ))
                
                feedback_id = cursor.lastrowid
                conn.commit()
                
                logging.info(f"Stored feedback: {food_class} -> {feedback} (ID: {feedback_id})")
                
        except Exception as e:
            raise RuntimeError(f"Failed to store feedback: {str(e)}")
    
    def get_user_images(self, processed_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all user images from database.
        
        Args:
            processed_only: Only return images with processed embeddings
            
        Returns:
            List of user image records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()
                
                query = '''
                    SELECT id, filename, original_name, upload_timestamp, 
                           embedding_path, processed, file_size_bytes,
                           image_width, image_height
                    FROM user_images
                '''
                
                if processed_only:
                    query += ' WHERE processed = TRUE'
                
                query += ' ORDER BY upload_timestamp DESC'
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            raise RuntimeError(f"Failed to get user images: {str(e)}")
    
    def get_user_embeddings(self) -> List[Tuple[int, str]]:
        """
        Get all user images with processed embeddings.
        
        Returns:
            List of tuples (image_id, embedding_path)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, embedding_path
                    FROM user_images
                    WHERE processed = TRUE AND embedding_path IS NOT NULL
                    ORDER BY upload_timestamp DESC
                ''')
                
                return cursor.fetchall()
                
        except Exception as e:
            raise RuntimeError(f"Failed to get user embeddings: {str(e)}")
    
    def get_feedback_for_food(self, food_class: str) -> List[Dict[str, Any]]:
        """
        Get all feedback for a specific food class.
        
        Args:
            food_class: Food class to get feedback for
            
        Returns:
            List of feedback records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT f.*, ui.filename, ui.original_name
                    FROM recommendation_feedback f
                    JOIN user_images ui ON f.user_image_id = ui.id
                    WHERE f.food_class = ?
                    ORDER BY f.feedback_timestamp DESC
                ''', (food_class,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            raise RuntimeError(f"Failed to get feedback for food: {str(e)}")
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get overall feedback statistics.
        
        Returns:
            Dictionary with feedback statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Overall feedback counts
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_feedback,
                        SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) as likes,
                        SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END) as dislikes,
                        SUM(CASE WHEN feedback = 0 THEN 1 ELSE 0 END) as neutral
                    FROM recommendation_feedback
                ''')
                
                overall_stats = dict(zip(['total_feedback', 'likes', 'dislikes', 'neutral'], 
                                       cursor.fetchone()))
                
                # Top liked foods
                cursor.execute('''
                    SELECT food_class, food_name, COUNT(*) as like_count
                    FROM recommendation_feedback
                    WHERE feedback = 1
                    GROUP BY food_class, food_name
                    ORDER BY like_count DESC
                    LIMIT 10
                ''')
                
                top_liked = [dict(zip(['food_class', 'food_name', 'like_count'], row))
                           for row in cursor.fetchall()]
                
                # Most disliked foods
                cursor.execute('''
                    SELECT food_class, food_name, COUNT(*) as dislike_count
                    FROM recommendation_feedback
                    WHERE feedback = -1
                    GROUP BY food_class, food_name
                    ORDER BY dislike_count DESC
                    LIMIT 10
                ''')
                
                top_disliked = [dict(zip(['food_class', 'food_name', 'dislike_count'], row))
                              for row in cursor.fetchall()]
                
                return {
                    **overall_stats,
                    'top_liked': top_liked,
                    'top_disliked': top_disliked
                }
                
        except Exception as e:
            raise RuntimeError(f"Failed to get feedback stats: {str(e)}")
    
    def create_session(self, session_id: str) -> str:
        """
        Create a new user session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO user_sessions
                    (session_id, start_timestamp, num_uploads, num_recommendations)
                    VALUES (?, ?, 0, 0)
                ''', (session_id, datetime.now()))
                
                conn.commit()
                logging.info(f"Created session: {session_id}")
                return session_id
                
        except Exception as e:
            raise RuntimeError(f"Failed to create session: {str(e)}")
    
    def export_data(self, output_path: str):
        """
        Export all data to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        try:
            data = {
                'user_images': self.get_user_images(),
                'feedback_stats': self.get_feedback_stats(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logging.info(f"Exported data to {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to export data: {str(e)}")
    
    def clear_user_data(self, confirm: bool = False):
        """
        Clear all user data (DANGEROUS - for testing only).
        
        Args:
            confirm: Must be True to actually clear data
        """
        if not confirm:
            raise ValueError("Must confirm to clear user data")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM recommendation_feedback')
                cursor.execute('DELETE FROM user_images')
                cursor.execute('DELETE FROM user_sessions')
                
                conn.commit()
                logging.warning("Cleared all user data from database")
                
        except Exception as e:
            raise RuntimeError(f"Failed to clear user data: {str(e)}")
    
    def delete_user_image(self, image_id: int) -> Dict[str, Any]:
        """
        Delete a user image and its associated data.
        
        Args:
            image_id: Database ID of the image to delete
            
        Returns:
            Dictionary with deletion result and file paths for cleanup
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get image info before deletion
                cursor.execute('''
                    SELECT filename, embedding_path, original_name
                    FROM user_images 
                    WHERE id = ?
                ''', (image_id,))
                
                image_info = cursor.fetchone()
                if not image_info:
                    return {
                        "success": False,
                        "error": f"Image with ID {image_id} not found"
                    }
                
                image_info = dict(image_info)
                
                # Delete associated feedback first (foreign key constraint)
                cursor.execute('''
                    DELETE FROM recommendation_feedback 
                    WHERE user_image_id = ?
                ''', (image_id,))
                
                feedback_deleted = cursor.rowcount
                
                # Delete the image record
                cursor.execute('''
                    DELETE FROM user_images 
                    WHERE id = ?
                ''', (image_id,))
                
                if cursor.rowcount == 0:
                    return {
                        "success": False,
                        "error": f"Failed to delete image record {image_id}"
                    }
                
                conn.commit()
                
                return {
                    "success": True,
                    "image_info": image_info,
                    "feedback_deleted": feedback_deleted,
                    "message": f"Deleted image '{image_info['original_name']}' and {feedback_deleted} associated feedback records"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete image: {str(e)}"
            }
    
    def get_user_image_details(self, image_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific user image.
        
        Args:
            image_id: Database ID of the image
            
        Returns:
            Dictionary with image details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get image details
                cursor.execute('''
                    SELECT id, filename, original_name, upload_timestamp, 
                           embedding_path, processed, file_size_bytes,
                           image_width, image_height
                    FROM user_images
                    WHERE id = ?
                ''', (image_id,))
                
                image_row = cursor.fetchone()
                if not image_row:
                    return {
                        "success": False,
                        "error": f"Image with ID {image_id} not found"
                    }
                
                image_details = dict(image_row)
                
                # Get associated feedback
                cursor.execute('''
                    SELECT food_class, food_name, feedback, similarity_score, 
                           feedback_timestamp
                    FROM recommendation_feedback
                    WHERE user_image_id = ?
                    ORDER BY feedback_timestamp DESC
                ''', (image_id,))
                
                feedback_rows = cursor.fetchall()
                feedback_list = [dict(row) for row in feedback_rows]
                
                return {
                    "success": True,
                    "image_details": image_details,
                    "feedback_history": feedback_list,
                    "feedback_count": len(feedback_list)
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get image details: {str(e)}"
            }

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count records in each table
                cursor.execute('SELECT COUNT(*) FROM user_images')
                num_images = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM user_images WHERE processed = TRUE')
                num_processed = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM recommendation_feedback')
                num_feedback = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM user_sessions')
                num_sessions = cursor.fetchone()[0]
                
                # Database file size
                db_size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'database_path': self.db_path,
                    'database_size_bytes': db_size_bytes,
                    'num_user_images': num_images,
                    'num_processed_images': num_processed,
                    'num_feedback_records': num_feedback,
                    'num_sessions': num_sessions
                }
                
        except Exception as e:
            raise RuntimeError(f"Failed to get database stats: {str(e)}")


def test_database_manager():
    """Test function for the DatabaseManager class."""
    print("Testing DatabaseManager...")
    
    try:
        # Use test database
        test_db_path = "app_data/test_database.db"
        
        # Clean up any existing test database
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        # Initialize database manager
        db = DatabaseManager(test_db_path)
        print(f"✓ DatabaseManager initialized: {db.get_database_stats()}")
        
        # Test storing user image
        image_id = db.store_user_image(
            filename="test_image_001.jpg",
            original_name="my_food_photo.jpg",
            file_size_bytes=1024,
            image_width=800,
            image_height=600
        )
        print(f"✓ Stored user image with ID: {image_id}")
        
        # Test updating embedding
        embedding_path = "app_data/user_embeddings/test_image_001.npy"
        db.update_image_embedding(image_id, embedding_path)
        print("✓ Updated image embedding path")
        
        # Test storing feedback
        db.store_feedback(
            user_image_id=image_id,
            food_class="apple_pie",
            food_name="Apple Pie",
            feedback=1,
            similarity_score=0.85
        )
        print("✓ Stored feedback")
        
        # Test data retrieval
        user_images = db.get_user_images()
        print(f"✓ Retrieved {len(user_images)} user images")
        
        feedback_stats = db.get_feedback_stats()
        print(f"✓ Retrieved feedback stats: {feedback_stats['total_feedback']} total feedback")
        
        # Test session creation
        session_id = "test_session_123"
        db.create_session(session_id)
        print(f"✓ Created session: {session_id}")
        
        # Final stats
        final_stats = db.get_database_stats()
        print(f"✓ Final database stats: {final_stats}")
        
        # Cleanup
        db.clear_user_data(confirm=True)
        print("✓ Cleaned up test data")
        
        # Remove test database
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        print("DatabaseManager test completed successfully!")
        
    except Exception as e:
        print(f"✗ DatabaseManager test failed: {str(e)}")


if __name__ == "__main__":
    test_database_manager()