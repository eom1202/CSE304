"""
Multimodal dataset loading and management
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
from typing import List, Dict, Tuple, Any
import os
from sklearn.model_selection import train_test_split

class MultimodalDataset:
    """Multimodal dataset management for clustering experiments"""
    
    def __init__(self, source: str = 'synthetic', size: int = 500, config: Dict[str, Any] = None):
        """
        Initialize multimodal dataset
        
        Args:
            source: 'mscoco' | 'synthetic' | 'flickr30k'
            size: Number of samples
            config: Configuration dictionary
        """
        self.source = source
        self.size = size
        self.config = config or {}
        self.data = []
        self.categories = self.config.get('categories', ['person', 'vehicle', 'animal', 'furniture', 'food'])
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load data based on source"""
        if self.source == 'mscoco':
            return self.load_mscoco_subset()
        elif self.source == 'synthetic':
            return self.create_synthetic_realistic()
        else:
            raise ValueError(f"Unsupported data source: {self.source}")
    
    def load_mscoco_subset(self) -> List[Dict[str, Any]]:
        """
        Load MS-COCO subset with balanced category sampling
        """
        try:
            from pycocotools.coco import COCO
            import requests
            from io import BytesIO
            import json
        except ImportError:
            print("Error: pycocotools not installed. Install with: pip install pycocotools")
            print("Falling back to synthetic data...")
            return self.create_synthetic_realistic()
        
        # MS-COCO dataset paths (you need to download these)
        coco_data_dir = self.config.get('coco_data_dir', './data/coco')
        annotations_file = os.path.join(coco_data_dir, 'annotations/instances_val2017.json')
        captions_file = os.path.join(coco_data_dir, 'annotations/captions_val2017.json')
        images_dir = os.path.join(coco_data_dir, 'val2017')
        
        # Check if files exist
        if not all(os.path.exists(f) for f in [annotations_file, captions_file, images_dir]):
            print(f"MS-COCO data not found in {coco_data_dir}")
            print("Please download MS-COCO dataset or use synthetic data")
            print("Falling back to synthetic data...")
            return self.create_synthetic_realistic()
        
        print(f"Loading MS-COCO data from {coco_data_dir}...")
        
        # Initialize COCO APIs
        coco_instances = COCO(annotations_file)
        coco_captions = COCO(captions_file)
        
        # Define category mapping to our target categories
        category_mapping = self._get_coco_category_mapping(coco_instances)
        
        # Sample images for each target category
        sampled_data = []
        samples_per_category = self.size // len(self.categories)
        
        for target_category in self.categories:
            if target_category not in category_mapping:
                print(f"Warning: No COCO categories mapped to {target_category}")
                continue
                
            coco_cat_ids = category_mapping[target_category]
            
            # Get images containing these categories
            img_ids = set()
            for cat_id in coco_cat_ids:
                img_ids.update(coco_instances.getImgIds(catIds=[cat_id]))
            
            img_ids = list(img_ids)
            
            if len(img_ids) == 0:
                print(f"No images found for category {target_category}")
                continue
            
            # Sample images
            n_samples = min(samples_per_category, len(img_ids))
            sampled_img_ids = np.random.choice(img_ids, size=n_samples, replace=False)
            
            for img_id in sampled_img_ids:
                try:
                    # Load image
                    img_info = coco_instances.loadImgs([img_id])[0]
                    image_path = os.path.join(images_dir, img_info['file_name'])
                    
                    if not os.path.exists(image_path):
                        continue
                    
                    image = Image.open(image_path).convert('RGB')
                    
                    # Get captions
                    caption_ids = coco_captions.getAnnIds(imgIds=[img_id])
                    captions = coco_captions.loadAnns(caption_ids)
                    
                    if not captions:
                        continue
                    
                    # Use the first caption or combine multiple captions
                    caption_text = captions[0]['caption']
                    if len(captions) > 1:
                        # Combine multiple captions
                        all_captions = [cap['caption'] for cap in captions[:3]]  # Use up to 3 captions
                        caption_text = '. '.join(all_captions)
                    
                    sampled_data.append({
                        'image': image,
                        'caption': caption_text,
                        'category': target_category,
                        'sample_id': f"coco_{target_category}_{img_id}",
                        'coco_img_id': img_id,
                        'original_filename': img_info['file_name']
                    })
                    
                except Exception as e:
                    print(f"Error loading image {img_id}: {e}")
                    continue
        
        if len(sampled_data) < self.size // 2:
            print(f"Warning: Only loaded {len(sampled_data)} samples from COCO")
            print("Consider using synthetic data or checking COCO dataset")
        
        # Shuffle and limit to requested size
        np.random.shuffle(sampled_data)
        sampled_data = sampled_data[:self.size]
        
        self.data = sampled_data
        print(f"Loaded {len(sampled_data)} samples from MS-COCO dataset")
        
        return sampled_data
    
    def _get_coco_category_mapping(self, coco_instances) -> Dict[str, List[int]]:
        """
        Map our target categories to COCO category IDs
        """
        # Get all COCO categories
        coco_categories = coco_instances.loadCats(coco_instances.getCatIds())
        coco_cat_dict = {cat['name'].lower(): cat['id'] for cat in coco_categories}
        
        # Define mapping from our categories to COCO categories
        category_mapping = {
            'person': [],
            'vehicle': [],
            'animal': [],
            'furniture': [],
            'food': []
        }
        
        # Person category
        person_keywords = ['person']
        for keyword in person_keywords:
            if keyword in coco_cat_dict:
                category_mapping['person'].append(coco_cat_dict[keyword])
        
        # Vehicle category
        vehicle_keywords = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'airplane', 'train', 'boat']
        for keyword in vehicle_keywords:
            if keyword in coco_cat_dict:
                category_mapping['vehicle'].append(coco_cat_dict[keyword])
        
        # Animal category
        animal_keywords = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        for keyword in animal_keywords:
            if keyword in coco_cat_dict:
                category_mapping['animal'].append(coco_cat_dict[keyword])
        
        # Furniture category
        furniture_keywords = ['chair', 'couch', 'bed', 'dining table', 'toilet']
        for keyword in furniture_keywords:
            if keyword in coco_cat_dict:
                category_mapping['furniture'].append(coco_cat_dict[keyword])
        
        # Food category
        food_keywords = ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']
        for keyword in food_keywords:
            if keyword in coco_cat_dict:
                category_mapping['food'].append(coco_cat_dict[keyword])
        
        # Print mapping info
        print("COCO Category Mapping:")
        for target_cat, coco_ids in category_mapping.items():
            coco_names = [cat['name'] for cat in coco_categories if cat['id'] in coco_ids]
            print(f"  {target_cat}: {coco_names}")
        
        return category_mapping
    
    def create_synthetic_realistic(self) -> List[Dict[str, Any]]:
        """
        Create sophisticated synthetic multimodal data
        """
        print(f"Creating {self.size} synthetic multimodal samples...")
        
        # Define category-specific patterns
        category_patterns = {
            'person': {
                'text_keywords': ['person', 'man', 'woman', 'people', 'human', 'individual', 'standing', 'walking'],
                'image_color': (255, 200, 150),  # Skin tone
                'shapes': ['circle', 'oval']  # Head shapes
            },
            'vehicle': {
                'text_keywords': ['car', 'truck', 'vehicle', 'automobile', 'driving', 'road', 'transportation'],
                'image_color': (100, 100, 200),  # Blue-ish
                'shapes': ['rectangle', 'rounded_rectangle']
            },
            'animal': {
                'text_keywords': ['dog', 'cat', 'animal', 'pet', 'wildlife', 'creature', 'furry', 'tail'],
                'image_color': (150, 100, 50),  # Brown-ish
                'shapes': ['circle', 'oval', 'irregular']
            },
            'furniture': {
                'text_keywords': ['chair', 'table', 'furniture', 'wooden', 'sitting', 'room', 'home'],
                'image_color': (139, 69, 19),  # Brown
                'shapes': ['rectangle', 'square']
            },
            'food': {
                'text_keywords': ['food', 'meal', 'eating', 'delicious', 'kitchen', 'cooking', 'fresh'],
                'image_color': (255, 165, 0),  # Orange-ish
                'shapes': ['circle', 'irregular']
            }
        }
        
        samples_per_category = self.size // len(self.categories)
        data = []
        
        for category in self.categories:
            pattern = category_patterns[category]
            
            for i in range(samples_per_category):
                # Generate text
                text = self._generate_category_text(pattern['text_keywords'], category)
                
                # Generate image
                image = self._generate_category_image(pattern['image_color'], pattern['shapes'])
                
                # Add some cross-modal noise for realism
                if random.random() < 0.1:  # 10% noise
                    # Sometimes mix categories slightly
                    noise_category = random.choice([c for c in self.categories if c != category])
                    noise_pattern = category_patterns[noise_category]
                    text += f" near {random.choice(noise_pattern['text_keywords'])}"
                
                data.append({
                    'image': image,
                    'caption': text,
                    'category': category,
                    'sample_id': f"{category}_{i:03d}"
                })
        
        # Add remaining samples to balance
        remaining = self.size - len(data)
        for i in range(remaining):
            category = random.choice(self.categories)
            pattern = category_patterns[category]
            
            text = self._generate_category_text(pattern['text_keywords'], category)
            image = self._generate_category_image(pattern['image_color'], pattern['shapes'])
            
            data.append({
                'image': image,
                'caption': text,
                'category': category,
                'sample_id': f"{category}_extra_{i:03d}"
            })
        
        # Shuffle data
        random.shuffle(data)
        self.data = data
        
        print(f"Created {len(data)} synthetic samples across {len(self.categories)} categories")
        return data
    
    def _generate_category_text(self, keywords: List[str], category: str) -> str:
        """Generate realistic text for a category"""
        # Base description
        main_keyword = random.choice(keywords)
        
        # Add descriptive words
        descriptors = {
            'person': ['tall', 'young', 'smiling', 'happy', 'standing'],
            'vehicle': ['red', 'fast', 'modern', 'parked', 'moving'],
            'animal': ['cute', 'small', 'playful', 'brown', 'running'],
            'furniture': ['wooden', 'comfortable', 'modern', 'large', 'elegant'],
            'food': ['fresh', 'delicious', 'colorful', 'healthy', 'tasty']
        }
        
        descriptor = random.choice(descriptors.get(category, ['nice', 'good', 'interesting']))
        
        # Create sentence patterns
        patterns = [
            f"A {descriptor} {main_keyword} in the scene",
            f"This image shows a {descriptor} {main_keyword}",
            f"Picture of a {main_keyword} that looks {descriptor}",
            f"A {main_keyword} appears {descriptor} in this photo"
        ]
        
        base_text = random.choice(patterns)
        
        # Add additional context sometimes
        if random.random() < 0.3:
            additional_keywords = [k for k in keywords if k != main_keyword]
            if additional_keywords:
                additional = random.choice(additional_keywords)
                base_text += f" with {additional}"
        
        return base_text
    
    def _generate_category_image(self, base_color: Tuple[int, int, int], shapes: List[str]) -> Image.Image:
        """Generate synthetic image for a category"""
        # Create image
        img_size = (224, 224)
        image = Image.new('RGB', img_size, color=(240, 240, 240))  # Light gray background
        draw = ImageDraw.Draw(image)
        
        # Add some background texture
        for _ in range(20):
            x, y = random.randint(0, img_size[0]), random.randint(0, img_size[1])
            color_noise = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in (200, 200, 200))
            draw.point((x, y), fill=color_noise)
        
        # Draw main shape
        shape = random.choice(shapes)
        
        # Add color variation
        color_variation = tuple(max(0, min(255, c + random.randint(-50, 50))) for c in base_color)
        
        # Shape parameters
        center_x, center_y = img_size[0] // 2, img_size[1] // 2
        size_factor = random.uniform(0.3, 0.7)
        width = int(img_size[0] * size_factor)
        height = int(img_size[1] * size_factor)
        
        left = center_x - width // 2
        top = center_y - height // 2
        right = center_x + width // 2
        bottom = center_y + height // 2
        
        if shape == 'circle':
            draw.ellipse([left, top, right, bottom], fill=color_variation)
        elif shape == 'oval':
            # Slightly elongated
            draw.ellipse([left, top + height//4, right, bottom - height//4], fill=color_variation)
        elif shape == 'rectangle':
            draw.rectangle([left, top, right, bottom], fill=color_variation)
        elif shape == 'square':
            size = min(width, height)
            left = center_x - size // 2
            top = center_y - size // 2
            draw.rectangle([left, top, left + size, top + size], fill=color_variation)
        elif shape == 'rounded_rectangle':
            # Approximate rounded rectangle with multiple rectangles
            corner_radius = min(width, height) // 8
            draw.rectangle([left + corner_radius, top, right - corner_radius, bottom], fill=color_variation)
            draw.rectangle([left, top + corner_radius, right, bottom - corner_radius], fill=color_variation)
            draw.ellipse([left, top, left + 2*corner_radius, top + 2*corner_radius], fill=color_variation)
            draw.ellipse([right - 2*corner_radius, top, right, top + 2*corner_radius], fill=color_variation)
            draw.ellipse([left, bottom - 2*corner_radius, left + 2*corner_radius, bottom], fill=color_variation)
            draw.ellipse([right - 2*corner_radius, bottom - 2*corner_radius, right, bottom], fill=color_variation)
        elif shape == 'irregular':
            # Draw multiple overlapping circles for irregular shape
            for _ in range(3):
                offset_x = random.randint(-width//4, width//4)
                offset_y = random.randint(-height//4, height//4)
                size = random.randint(width//3, width//2)
                draw.ellipse([
                    center_x + offset_x - size//2,
                    center_y + offset_y - size//2,
                    center_x + offset_x + size//2,
                    center_y + offset_y + size//2
                ], fill=color_variation)
        
        # Add some details/noise
        for _ in range(5):
            x = random.randint(left, right)
            y = random.randint(top, bottom)
            detail_color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in color_variation)
            detail_size = random.randint(2, 8)
            draw.ellipse([x-detail_size, y-detail_size, x+detail_size, y+detail_size], fill=detail_color)
        
        return image
    
    def get_balanced_split(self, ratios: Dict[str, float] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split data maintaining class balance
        
        Args:
            ratios: Dictionary with 'train' and 'test' ratios
            
        Returns:
            Dictionary with 'train' and 'test' splits
        """
        if ratios is None:
            ratios = {'train': 0.7, 'test': 0.3}
        
        if not self.data:
            self.load_data()
        
        # Group by category
        category_data = {}
        for sample in self.data:
            category = sample['category']
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(sample)
        
        train_data = []
        test_data = []
        
        # Split each category separately
        for category, samples in category_data.items():
            train_samples, test_samples = train_test_split(
                samples, 
                train_size=ratios['train'],
                random_state=42,
                shuffle=True
            )
            train_data.extend(train_samples)
            test_data.extend(test_samples)
        
        # Shuffle final splits
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        print(f"Data split: {len(train_data)} train, {len(test_data)} test samples")
        
        return {
            'train': train_data,
            'test': test_data
        }
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.data:
            self.load_data()
        
        category_counts = {}
        text_lengths = []
        
        for sample in self.data:
            category = sample['category']
            category_counts[category] = category_counts.get(category, 0) + 1
            text_lengths.append(len(sample['caption'].split()))
        
        return {
            'total_samples': len(self.data),
            'categories': list(category_counts.keys()),
            'category_distribution': category_counts,
            'avg_text_length': np.mean(text_lengths),
            'text_length_std': np.std(text_lengths),
            'image_size': (224, 224)  # Fixed for synthetic data
        } 