#!/usr/bin/env python3
"""
Food Recommendation App Launcher
Simple script to check system status and launch the Streamlit app
"""
import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = {
        'streamlit': 'streamlit', 
        'torch': 'torch', 
        'torchvision': 'torchvision', 
        'faiss': 'faiss', 
        'numpy': 'numpy', 
        'pillow': 'PIL'  # Pillow is imported as PIL
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name.replace('-', '_'))
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def check_faiss_index():
    """Check if FAISS index exists."""
    index_path = Path("app_data/faiss_index.bin")
    if not index_path.exists():
        print("FAISS index not found!")
        print("Run setup first: python -m data.food101_processor --full-setup --max-images-per-class 10")
        return False
    return True

def check_food101_dataset():
    """Check if Food-101 dataset exists."""
    dataset_path = Path("food-101/images")
    if not dataset_path.exists():
        print("Food-101 dataset not found!")
        print("Please ensure the Food-101 dataset is in the 'food-101/' directory")
        return False
    return True

def main():
    """Main launcher function."""
    print("Food Recommendation App Launcher")
    print("=" * 40)
    
    # Fix OpenMP issue
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Check system requirements
    print("Checking system requirements...")
    
    if not check_dependencies():
        sys.exit(1)
    print("Dependencies OK")
    
    if not check_food101_dataset():
        sys.exit(1)
    print("Food-101 dataset found")
    
    if not check_faiss_index():
        response = input("\n🤔 Would you like to run the setup now? (y/n): ")
        if response.lower() in ['y', 'yes']:
            print("🔄 Running setup...")
            try:
                subprocess.run([
                    sys.executable, "-m", "data.food101_processor", 
                    "--full-setup", "--max-images-per-class", "10"
                ], check=True)
                print("Setup completed!")
            except subprocess.CalledProcessError:
                print("Setup failed!")
                sys.exit(1)
        else:
            sys.exit(1)
    else:
        print("FAISS index found")
    
    # Launch Streamlit app
    print("\nLaunching Streamlit app...")
    print("Open your browser to: http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print("-" * 40)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"], check=True)
    except KeyboardInterrupt:
        print("\nApp stopped. Thanks for using Food Recommendations!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()