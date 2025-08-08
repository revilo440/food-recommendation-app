"""
Configuration settings for the Food Recommendation App
"""
import os

# Model Configuration
MODEL_NAME = "resnet50"
EMBEDDING_SIZE = 2048
IMAGE_SIZE = (224, 224)

# Recommendation Settings
DEFAULT_NUM_RECOMMENDATIONS = 5
MIN_RECOMMENDATIONS = 1
MAX_RECOMMENDATIONS = 20
MIN_SIMILARITY_THRESHOLD = 0.1

# File Upload Settings
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png']
MAX_FILE_SIZE_MB = 15

# Performance Settings
PRELOAD_FAISS_INDEX = True
ENABLE_FEEDBACK = True
BATCH_SIZE = 32

# Paths
FOOD101_IMAGES_PATH = "food-101/images"
APP_DATA_PATH = "app_data"
USER_IMAGES_PATH = os.path.join(APP_DATA_PATH, "user_images")
USER_EMBEDDINGS_PATH = os.path.join(APP_DATA_PATH, "user_embeddings")
FOOD101_EMBEDDINGS_PATH = os.path.join(APP_DATA_PATH, "food101_embeddings")
DATABASE_PATH = os.path.join(APP_DATA_PATH, "app_database.db")
FAISS_INDEX_PATH = os.path.join(APP_DATA_PATH, "faiss_index.bin")
FOOD_NAME_MAPPING_PATH = os.path.join(APP_DATA_PATH, "food_name_mapping.json")

# Debug Settings
DEBUG_MODE = False
VERBOSE_LOGGING = DEBUG_MODE
SAVE_INTERMEDIATE_IMAGES = DEBUG_MODE
PROFILE_PERFORMANCE = DEBUG_MODE

# Environment-specific settings
ENVIRONMENT = os.getenv('APP_ENV', 'development')

if ENVIRONMENT == 'production':
    DEBUG_MODE = False
    BATCH_SIZE = 32
    PRELOAD_FAISS_INDEX = True
elif ENVIRONMENT == 'development':
    DEBUG_MODE = True
    BATCH_SIZE = 8
    PRELOAD_FAISS_INDEX = False