# Food Recommendation App

A local-first food recommendation system that learns from your visual preferences using computer vision and similarity matching.

## Features

- **Visual Food Recognition**: Upload food photos and get instant recommendations
- **Smart Similarity Search**: Uses ResNet50 + FAISS for fast, accurate matching
- **Learning System**: Like/dislike feedback improves future recommendations
- **Local-First**: All data stays private on your device
- **101 Food Categories**: Recommendations from curated Food-101 dataset
- **Minimalist UI**: Clean, responsive Streamlit interface

## Quick Start

### 1. Prerequisites
- Python 3.8+
- Food-101 dataset (should be in `food-101/` directory)

### 2. Setup
```bash
# Install dependencies (already done if you got here)
pip install torch torchvision faiss-cpu streamlit pillow numpy tqdm

# One-time preprocessing (creates embeddings for 1,010 sample images)
python -m data.food101_processor --full-setup --max-images-per-class 10
```

### 3. Run the App
```bash
streamlit run main.py
```

Open your browser to `http://localhost:8501`

## 📱 How to Use

1. **Upload**: Click "Choose a food image" and select a food photo
2. **Process**: Click "Get Recommendations" to analyze your image  
3. **Discover**: Browse recommended similar foods with similarity scores
4. **Feedback**: Use 👍/👎 buttons to train the system
5. **Manage**: Click "Manage Images" to view, delete, or get details about your uploads
6. **View**: Click on any image to see full-size view with metadata and feedback history
7. **Delete**: Remove unwanted images with confirmation (cleans up files and database)
8. **Explore**: Upload more images to build your preference profile

## Architecture

```
Project Structure
├── main.py                    # Streamlit web interface
├── config.py                  # Configuration settings
├── models/
│   ├── feature_extractor.py   # ResNet50 feature extraction
│   └── similarity_engine.py   # FAISS similarity search
├── data/
│   ├── database_manager.py    # SQLite data storage
│   └── food101_processor.py   # Dataset preprocessing
├── utils/
│   └── user_image_processor.py # User image pipeline
└── app_data/                  # Generated data
    ├── user_images/           # Your uploaded images
    ├── user_embeddings/       # Extracted features
    ├── food101_embeddings/    # Food-101 features
    ├── faiss_index.bin        # Similarity search index
    └── app_database.db        # SQLite database
```

## Configuration

Edit `config.py` to customize:
- Number of recommendations (1-20)
- Similarity thresholds
- Supported image formats
- File size limits
- Model parameters

## Commands

```bash
# Validate Food-101 dataset
python -m data.food101_processor --validate-dataset

# Rebuild embeddings (full dataset ~101k images)
python -m data.food101_processor --full-setup

# Build FAISS index only
python -m data.food101_processor --build-index-only

# Test system components
python test_complete_system.py
```

## Technology Stack

- **ML**: PyTorch, ResNet50, FAISS
- **UI**: Streamlit, HTML/CSS
- **Data**: SQLite, NumPy, PIL
- **Dataset**: Food-101 (101 categories, 101k images)
- **Features**: Image management, deletion, metadata viewing

## Included Files

This repository includes pre-computed embeddings and a FAISS index from the Food-101 dataset, allowing you to run the application without downloading the full 5GB dataset:

- Embeddings in `app_data/food101_embeddings/`
- FAISS similarity search index `app_data/faiss_index.bin`
- Metadata and class mappings

If you want to work with the full dataset or regenerate embeddings:
1. Download Food-101 from [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
2. Run `python -m data.food101_processor --full-setup`

## License

This project is for educational and research purposes.

## Acknowledgments

- Food-101 dataset by ETH Zurich
- PyTorch and torchvision teams
- FAISS by Facebook AI Research
- Streamlit development team

---

**Enjoy discovering new foods!**
