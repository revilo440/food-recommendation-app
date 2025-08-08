"""
Similarity Engine using FAISS for fast food recommendation search
"""
import numpy as np
import faiss
import json
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import logging

from config import (
    EMBEDDING_SIZE, DEFAULT_NUM_RECOMMENDATIONS, MIN_SIMILARITY_THRESHOLD,
    FOOD101_EMBEDDINGS_PATH, FAISS_INDEX_PATH, FOOD_NAME_MAPPING_PATH,
    PRELOAD_FAISS_INDEX
)


class SimilarityEngine:
    """
    Similarity engine for finding visually similar foods using FAISS.
    Handles Food-101 embeddings and provides fast similarity search.
    """
    
    def __init__(self, embedding_dir: str = FOOD101_EMBEDDINGS_PATH):
        """
        Initialize the similarity engine.
        
        Args:
            embedding_dir: Directory containing Food-101 embeddings
        """
        self.embedding_dir = embedding_dir
        self.index = None
        self.food_mappings = {}
        self.embedding_files = []
        self.food_classes = []
        
        logging.info(f"SimilarityEngine initialized with embedding directory: {embedding_dir}")
        
        # Load food name mappings if available
        self._load_food_mappings()
        
        # Pre-load index if configured
        if PRELOAD_FAISS_INDEX and os.path.exists(FAISS_INDEX_PATH):
            self.load_faiss_index()
    
    def _load_food_mappings(self):
        """Load food name mappings from JSON file."""
        try:
            if os.path.exists(FOOD_NAME_MAPPING_PATH):
                with open(FOOD_NAME_MAPPING_PATH, 'r') as f:
                    self.food_mappings = json.load(f)
                logging.info(f"Loaded {len(self.food_mappings)} food name mappings")
            else:
                logging.warning(f"Food name mapping file not found: {FOOD_NAME_MAPPING_PATH}")
        except Exception as e:
            logging.error(f"Failed to load food name mappings: {str(e)}")
            self.food_mappings = {}
    
    def _get_display_name(self, food_class: str) -> str:
        """
        Get user-friendly display name for a food class.
        
        Args:
            food_class: Snake case food class name (e.g., 'apple_pie')
            
        Returns:
            Title case display name (e.g., 'Apple Pie')
        """
        if food_class in self.food_mappings:
            return self.food_mappings[food_class]
        else:
            # Fallback: convert snake_case to Title Case
            return food_class.replace('_', ' ').title()
    
    def build_faiss_index(self, embeddings: np.ndarray, embedding_files: List[str]) -> faiss.Index:
        """
        Build FAISS index for fast similarity search.
        
        Args:
            embeddings: Array of shape (num_embeddings, embedding_size)
            embedding_files: List of embedding file names corresponding to embeddings
            
        Returns:
            Trained FAISS index
        """
        try:
            num_embeddings, embedding_dim = embeddings.shape
            
            if embedding_dim != EMBEDDING_SIZE:
                raise ValueError(f"Expected embedding dimension {EMBEDDING_SIZE}, got {embedding_dim}")
            
            # Create FAISS index for cosine similarity
            # Using IndexFlatIP (Inner Product) with L2 normalized vectors gives cosine similarity
            index = faiss.IndexFlatIP(embedding_dim)
            
            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            
            # Add embeddings to index
            index.add(embeddings_normalized.astype(np.float32))
            
            # Store metadata
            self.embedding_files = embedding_files
            self.food_classes = [self._extract_food_class(f) for f in embedding_files]
            
            logging.info(f"Built FAISS index with {num_embeddings} embeddings")
            return index
            
        except Exception as e:
            raise RuntimeError(f"Failed to build FAISS index: {str(e)}")
    
    def _extract_food_class(self, embedding_file: str) -> str:
        """Extract food class from embedding filename."""
        # Assuming embedding files are named like: apple_pie_001.npy
        filename = os.path.splitext(os.path.basename(embedding_file))[0]
        # Remove numeric suffix to get food class
        parts = filename.split('_')
        if parts[-1].isdigit():
            return '_'.join(parts[:-1])
        return filename
    
    def save_faiss_index(self, index_path: str = FAISS_INDEX_PATH):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save the index
        """
        try:
            if self.index is None:
                raise ValueError("No index to save. Build index first.")
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = index_path.replace('.bin', '_metadata.pkl')
            metadata = {
                'embedding_files': self.embedding_files,
                'food_classes': self.food_classes,
                'num_embeddings': self.index.ntotal
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logging.info(f"Saved FAISS index and metadata to {index_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save FAISS index: {str(e)}")
    
    def load_faiss_index(self, index_path: str = FAISS_INDEX_PATH):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to the saved index
        """
        try:
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = index_path.replace('.bin', '_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.embedding_files = metadata['embedding_files']
                self.food_classes = metadata['food_classes']
                
                logging.info(f"Loaded FAISS index with {self.index.ntotal} embeddings")
            else:
                logging.warning(f"Metadata file not found: {metadata_path}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {str(e)}")
    
    def aggregate_user_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate multiple user embeddings using simple averaging.
        
        Args:
            embeddings: List of user embedding vectors
            
        Returns:
            Aggregated embedding vector
        """
        if not embeddings:
            raise ValueError("No embeddings provided for aggregation")
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Simple average aggregation
        aggregated = np.mean(embeddings, axis=0)
        
        # Normalize the result
        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm
        
        return aggregated
    
    def find_similar_foods(self, user_embedding: np.ndarray, k: int = DEFAULT_NUM_RECOMMENDATIONS) -> List[Dict[str, Any]]:
        """
        Find similar foods to user embedding using FAISS search.
        
        Args:
            user_embedding: User's aggregated embedding vector
            k: Number of recommendations to return
            
        Returns:
            List of dictionaries with food recommendations
        """
        try:
            if self.index is None:
                raise ValueError("FAISS index not loaded. Load or build index first.")
            
            if user_embedding.shape != (EMBEDDING_SIZE,):
                raise ValueError(f"Expected embedding shape {(EMBEDDING_SIZE,)}, got {user_embedding.shape}")
            
            # Normalize user embedding for cosine similarity
            user_embedding_normalized = user_embedding.copy()
            norm = np.linalg.norm(user_embedding_normalized)
            if norm > 0:
                user_embedding_normalized = user_embedding_normalized / norm
            
            # Perform FAISS search
            user_embedding_normalized = user_embedding_normalized.astype(np.float32).reshape(1, -1)
            similarities, indices = self.index.search(user_embedding_normalized, k)
            
            # Format results
            recommendations = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= len(self.food_classes):
                    logging.warning(f"Invalid index {idx} for {len(self.food_classes)} food classes")
                    continue
                
                # Filter out very low similarity scores
                if similarity < MIN_SIMILARITY_THRESHOLD:
                    continue
                
                food_class = self.food_classes[idx]
                embedding_file = self.embedding_files[idx]
                
                # Get corresponding image path
                image_path = self._get_image_path_from_embedding(embedding_file)
                
                recommendation = {
                    'food_name': self._get_display_name(food_class),
                    'food_class': food_class,
                    'similarity_score': float(similarity),
                    'image_path': image_path,
                    'embedding_file': embedding_file,
                    'rank': i + 1
                }
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            raise RuntimeError(f"Similarity search failed: {str(e)}")
    
    def _get_image_path_from_embedding(self, embedding_file: str) -> str:
        """
        Get corresponding Food-101 image path from embedding filename.
        
        Args:
            embedding_file: Embedding filename (e.g., 'apple_pie_001.npy')
            
        Returns:
            Path to corresponding Food-101 image
        """
        # Extract food class and image ID
        basename = os.path.splitext(embedding_file)[0]
        parts = basename.split('_')
        
        if len(parts) >= 2 and parts[-1].isdigit():
            food_class = '_'.join(parts[:-1])
            image_id = parts[-1]
            
            # Construct image path
            image_filename = f"{image_id}.jpg"
            image_path = os.path.join("food-101", "images", food_class, image_filename)
            
            return image_path
        else:
            # Fallback if pattern doesn't match
            return f"food-101/images/{basename}.jpg"
    
    def search_by_multiple_embeddings(self, user_embeddings: List[np.ndarray], k: int = DEFAULT_NUM_RECOMMENDATIONS) -> List[Dict[str, Any]]:
        """
        Find similar foods using multiple user embeddings.
        
        Args:
            user_embeddings: List of user embedding vectors
            k: Number of recommendations to return
            
        Returns:
            List of food recommendations
        """
        # Aggregate embeddings
        aggregated_embedding = self.aggregate_user_embeddings(user_embeddings)
        
        # Find similar foods
        return self.find_similar_foods(aggregated_embedding, k)
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded FAISS index.
        
        Returns:
            Dictionary with index information
        """
        if self.index is None:
            return {"index_loaded": False}
        
        return {
            "index_loaded": True,
            "num_embeddings": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "index_type": type(self.index).__name__,
            "num_food_classes": len(set(self.food_classes)) if self.food_classes else 0,
            "food_mappings_loaded": len(self.food_mappings) > 0
        }


def test_similarity_engine():
    """Test function for the SimilarityEngine class."""
    print("Testing SimilarityEngine...")
    
    try:
        # Initialize engine
        engine = SimilarityEngine()
        print(f"✓ SimilarityEngine initialized: {engine.get_index_info()}")
        
        # Create dummy embeddings for testing
        print("Creating test embeddings...")
        num_test_embeddings = 100
        test_embeddings = np.random.randn(num_test_embeddings, EMBEDDING_SIZE).astype(np.float32)
        test_files = [f"test_food_{i:03d}.npy" for i in range(num_test_embeddings)]
        
        # Build index
        index = engine.build_faiss_index(test_embeddings, test_files)
        engine.index = index
        print(f"✓ Built test FAISS index: {engine.get_index_info()}")
        
        # Test similarity search
        query_embedding = np.random.randn(EMBEDDING_SIZE).astype(np.float32)
        results = engine.find_similar_foods(query_embedding, k=5)
        print(f"✓ Similarity search returned {len(results)} results")
        
        if results:
            print("  Top result:", {k: v for k, v in results[0].items() if k != 'embedding_file'})
        
        # Test aggregation
        multiple_embeddings = [np.random.randn(EMBEDDING_SIZE) for _ in range(3)]
        aggregated = engine.aggregate_user_embeddings(multiple_embeddings)
        print(f"✓ Aggregated {len(multiple_embeddings)} embeddings: shape={aggregated.shape}")
        
        print("SimilarityEngine test completed successfully!")
        
    except Exception as e:
        print(f"✗ SimilarityEngine test failed: {str(e)}")


if __name__ == "__main__":
    test_similarity_engine()