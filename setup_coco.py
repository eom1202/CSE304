#!/usr/bin/env python3
"""
MS-COCO Dataset Setup Script
Downloads and prepares MS-COCO dataset for multimodal clustering experiments
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import argparse

def download_file(url: str, filepath: str, description: str = ""):
    """Download a file with progress bar"""
    print(f"Downloading {description}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print(f"\n✓ Downloaded {description}")

def extract_zip(zip_path: str, extract_to: str, description: str = ""):
    """Extract zip file"""
    print(f"Extracting {description}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✓ Extracted {description}")

def setup_coco_dataset(data_dir: str = "./data/coco", subset: str = "val2017"):
    """
    Setup MS-COCO dataset for multimodal clustering
    
    Args:
        data_dir: Directory to store COCO data
        subset: COCO subset to download (val2017, train2017)
    """
    
    print("=" * 60)
    print("MS-COCO Dataset Setup for Multimodal Clustering")
    print("=" * 60)
    
    # Create directories
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    annotations_dir = data_path / "annotations"
    images_dir = data_path / subset
    
    annotations_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # COCO URLs
    base_url = "http://images.cocodataset.org"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    if subset == "val2017":
        images_url = f"{base_url}/zips/val2017.zip"
        subset_size = "~1GB"
    elif subset == "train2017":
        images_url = f"{base_url}/zips/train2017.zip"
        subset_size = "~18GB"
    else:
        raise ValueError(f"Unsupported subset: {subset}")
    
    print(f"Setting up COCO {subset} subset ({subset_size})")
    print(f"Data directory: {data_path.absolute()}")
    
    # Download annotations
    annotations_zip = data_path / "annotations_trainval2017.zip"
    if not annotations_zip.exists():
        download_file(annotations_url, str(annotations_zip), "annotations")
        extract_zip(str(annotations_zip), str(data_path), "annotations")
        annotations_zip.unlink()  # Remove zip file
    else:
        print("✓ Annotations already exist")
    
    # Download images
    images_zip = data_path / f"{subset}.zip"
    if not images_zip.exists() and not any(images_dir.iterdir()):
        print(f"\nWarning: {subset} images ({subset_size}) will be downloaded.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Skipping image download. You can download manually later.")
            print(f"Image URL: {images_url}")
        else:
            download_file(images_url, str(images_zip), f"{subset} images")
            extract_zip(str(images_zip), str(data_path), f"{subset} images")
            images_zip.unlink()  # Remove zip file
    else:
        print("✓ Images already exist or download skipped")
    
    # Verify setup
    print("\n" + "=" * 60)
    print("Verifying COCO dataset setup...")
    
    required_files = [
        annotations_dir / "instances_val2017.json",
        annotations_dir / "captions_val2017.json"
    ]
    
    all_good = True
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ {file_path.name}")
        else:
            print(f"✗ {file_path.name} - MISSING")
            all_good = False
    
    # Check for some images
    image_files = list(images_dir.glob("*.jpg"))
    if len(image_files) > 0:
        print(f"✓ Found {len(image_files)} images in {subset}/")
    else:
        print(f"✗ No images found in {subset}/ - Download may be needed")
        all_good = False
    
    if all_good:
        print("\n🎉 COCO dataset setup complete!")
        print(f"You can now use 'mscoco' as the data source in config.yaml")
        print(f"Set coco_data_dir: '{data_dir}' in your config")
    else:
        print("\n⚠️  Setup incomplete. Please check missing files.")
    
    return all_good

def create_sample_config():
    """Create a sample config file for COCO dataset"""
    config_content = """# Sample configuration for MS-COCO dataset

dataset:
  source: "mscoco"  # Use MS-COCO dataset
  n_samples: 500
  categories: ["person", "vehicle", "animal", "furniture", "food"]
  samples_per_category: 100
  train_ratio: 0.7
  test_ratio: 0.3
  
  # MS-COCO specific settings
  coco_data_dir: "./data/coco"  # Path to MS-COCO dataset
  coco_download_images: false   # Whether to download missing images
  coco_max_captions_per_image: 3  # Maximum captions to combine per image

# Rest of configuration...
# (Copy from config.yaml and modify as needed)
"""
    
    with open("config_coco.yaml", "w") as f:
        f.write(config_content)
    
    print("✓ Created config_coco.yaml sample configuration")

def main():
    parser = argparse.ArgumentParser(description="Setup MS-COCO dataset for multimodal clustering")
    parser.add_argument("--data-dir", default="./data/coco", 
                       help="Directory to store COCO data (default: ./data/coco)")
    parser.add_argument("--subset", default="val2017", choices=["val2017", "train2017"],
                       help="COCO subset to download (default: val2017)")
    parser.add_argument("--create-config", action="store_true",
                       help="Create sample configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    try:
        setup_coco_dataset(args.data_dir, args.subset)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 