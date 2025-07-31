"""
Visual Quality Spot-Check Generator
Creates comprehensive visual verification of data quality with random sampling.
"""

import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import seaborn as sns

class VisualQualityChecker:
    """Generate visual spot-checks for data quality verification."""
    
    def __init__(self, project_root: Path = None):
        """Initialize with project paths."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
            
        self.project_root = project_root
        self.processed_dir = project_root / "dataset" / "processed"
        self.splits_dir = project_root / "dataset" / "splits"
        self.results_dir = project_root / "results"
        
        # Create results directory
        self.visual_check_dir = self.results_dir / "visual_quality_checks"
        self.visual_check_dir.mkdir(parents=True, exist_ok=True)
        
    def load_splits_data(self) -> Dict[str, Any]:
        """Load all split data."""
        splits_data = {}
        
        for split_file in self.splits_dir.glob("*.json"):
            dataset_name = split_file.stem.replace("_splits", "")
            with open(split_file, 'r') as f:
                splits_data[dataset_name] = json.load(f)
                
        return splits_data
    
    def sample_images_from_split(self, dataset_name: str, split_type: str, 
                                n_samples: int = 6) -> List[str]:
        """Sample random images from a specific split."""
        splits_file = self.splits_dir / f"{dataset_name}_splits.json"
        
        if not splits_file.exists():
            return []
            
        with open(splits_file, 'r') as f:
            split_data = json.load(f)
            
        if 'splits' in split_data and split_type in split_data['splits']:
            available_ids = split_data['splits'][split_type]
            n_samples = min(n_samples, len(available_ids))
            return random.sample(available_ids, n_samples)
        
        return []
    
    def create_classification_spot_check(self, dataset_name: str) -> None:
        """Create visual spot-check for classification datasets."""
        print(f"Creating classification spot-check for {dataset_name}...")
        
        # Sample from train and val
        train_samples = self.sample_images_from_split(dataset_name, 'train', 3)
        val_samples = self.sample_images_from_split(dataset_name, 'val', 3)
        
        if not train_samples and not val_samples:
            print(f"  No samples found for {dataset_name}")
            return
            
        # Determine image directory
        if dataset_name == 'aptos2019':
            images_dir = self.processed_dir / "aptos2019" / "images"
        elif dataset_name == 'idrid_grading':
            images_dir = self.processed_dir / "grading" / "images"
        else:
            print(f"  Unknown dataset: {dataset_name}")
            return
            
        # Load grade information
        splits_file = self.splits_dir / f"{dataset_name}_splits.json"
        with open(splits_file, 'r') as f:
            split_data = json.load(f)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{dataset_name.upper()} - Classification Quality Spot-Check', 
                    fontsize=16, fontweight='bold')
        
        # Plot train samples
        for i, img_id in enumerate(train_samples):
            ax = axes[0, i]
            img_path = images_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                try:
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    ax.imshow(img_rgb)
                    ax.set_title(f'Train: {img_id}\\nShape: {img_rgb.shape}')
                    ax.axis('off')
                    
                    # Add image statistics
                    mean_val = np.mean(img_rgb)
                    std_val = np.std(img_rgb)
                    ax.text(0.02, 0.98, f'μ={mean_val:.1f}, σ={std_val:.1f}', 
                           transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor="white", alpha=0.8), fontsize=8, 
                           verticalalignment='top')
                           
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error loading\\n{img_id}\\n{str(e)}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Train: {img_id} (ERROR)')
            else:
                ax.text(0.5, 0.5, f'File not found:\\n{img_id}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Train: {img_id} (MISSING)')
            
        # Plot val samples
        for i, img_id in enumerate(val_samples):
            ax = axes[1, i]
            img_path = images_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                try:
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    ax.imshow(img_rgb)
                    ax.set_title(f'Val: {img_id}\\nShape: {img_rgb.shape}')
                    ax.axis('off')
                    
                    mean_val = np.mean(img_rgb)
                    std_val = np.std(img_rgb)
                    ax.text(0.02, 0.98, f'μ={mean_val:.1f}, σ={std_val:.1f}', 
                           transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor="white", alpha=0.8), fontsize=8, 
                           verticalalignment='top')
                           
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error loading\\n{img_id}\\n{str(e)}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Val: {img_id} (ERROR)')
            else:
                ax.text(0.5, 0.5, f'File not found:\\n{img_id}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Val: {img_id} (MISSING)')
        
        plt.tight_layout()
        save_path = self.visual_check_dir / f"{dataset_name}_classification_spot_check.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved classification spot-check to {save_path}")
    
    def create_segmentation_spot_check(self, dataset_name: str = 'idrid_segmentation') -> None:
        """Create visual spot-check for segmentation dataset."""
        print(f"Creating segmentation spot-check for {dataset_name}...")
        
        # Sample from train and val
        train_samples = self.sample_images_from_split(dataset_name, 'train', 2)
        val_samples = self.sample_images_from_split(dataset_name, 'val', 2)
        
        if not train_samples and not val_samples:
            print(f"  No samples found for {dataset_name}")
            return
            
        images_dir = self.processed_dir / "segmentation" / "images"
        masks_dir = self.processed_dir / "segmentation" / "masks"
        
        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('IDRiD Segmentation - Image/Mask Quality Spot-Check', 
                    fontsize=16, fontweight='bold')
        
        all_samples = [(sample, 'train') for sample in train_samples] + \
                     [(sample, 'val') for sample in val_samples]
        
        for idx, (img_id, split_type) in enumerate(all_samples):
            if idx >= 4:  # Limit to 4 samples
                break
                
            row = idx
            
            # Original image
            ax_img = axes[row, 0]
            img_path = images_dir / f"{img_id}.jpg"
            
            if img_path.exists():
                try:
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax_img.imshow(img_rgb)
                    ax_img.set_title(f'{split_type.upper()}: {img_id}\\nOriginal Image')
                    ax_img.axis('off')
                except Exception as e:
                    ax_img.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                    ax_img.set_title(f'{split_type.upper()}: {img_id} (ERROR)')
            else:
                ax_img.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                ax_img.set_title(f'{split_type.upper()}: {img_id} (MISSING)')
            
            # Mask
            ax_mask = axes[row, 1]
            mask_path = masks_dir / f"{img_id}.tif"
            
            if mask_path.exists():
                try:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    ax_mask.imshow(mask, cmap='gray')
                    ax_mask.set_title(f'Mask\\nUnique values: {len(np.unique(mask))}')
                    ax_mask.axis('off')
                    
                    # Add mask statistics
                    mask_mean = np.mean(mask)
                    mask_std = np.std(mask)
                    lesion_pixels = np.sum(mask > 0)
                    total_pixels = mask.size
                    lesion_ratio = lesion_pixels / total_pixels
                    
                    ax_mask.text(0.02, 0.98, 
                               f'Lesion: {lesion_ratio:.1%}\\nμ={mask_mean:.1f}', 
                               transform=ax_mask.transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                               fontsize=8, verticalalignment='top')
                               
                except Exception as e:
                    ax_mask.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                    ax_mask.set_title(f'Mask (ERROR)')
            else:
                ax_mask.text(0.5, 0.5, 'Mask not found', ha='center', va='center')
                ax_mask.set_title(f'Mask (MISSING)')
            
            # Overlay
            ax_overlay = axes[row, 2]
            if img_path.exists() and mask_path.exists():
                try:
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    
                    # Create colored mask overlay
                    colored_mask = np.zeros_like(img_rgb)
                    colored_mask[:, :, 0] = mask  # Red channel for lesions
                    
                    # Blend with original image
                    alpha = 0.4
                    overlay = cv2.addWeighted(img_rgb, 1-alpha, colored_mask, alpha, 0)
                    
                    ax_overlay.imshow(overlay)
                    ax_overlay.set_title('Image + Mask Overlay')
                    ax_overlay.axis('off')
                    
                except Exception as e:
                    ax_overlay.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                    ax_overlay.set_title('Overlay (ERROR)')
            else:
                ax_overlay.text(0.5, 0.5, 'Cannot create overlay', ha='center', va='center')
                ax_overlay.set_title('Overlay (MISSING DATA)')
                
            # Mask histogram
            ax_hist = axes[row, 3]
            if mask_path.exists():
                try:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    ax_hist.hist(mask.flatten(), bins=50, alpha=0.7, edgecolor='black')
                    ax_hist.set_title('Mask Value Distribution')
                    ax_hist.set_xlabel('Pixel Value')
                    ax_hist.set_ylabel('Frequency')
                    ax_hist.grid(True, alpha=0.3)
                except Exception as e:
                    ax_hist.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                    ax_hist.set_title('Histogram (ERROR)')
            else:
                ax_hist.text(0.5, 0.5, 'No mask data', ha='center', va='center')
                ax_hist.set_title('Histogram (NO DATA)')
        
        plt.tight_layout()
        save_path = self.visual_check_dir / f"{dataset_name}_spot_check.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved segmentation spot-check to {save_path}")
    
    def create_preprocessing_comparison(self) -> None:
        """Create before/after preprocessing comparison if raw data is available."""
        print("Creating preprocessing comparison...")
        
        # This would require access to raw data for comparison
        # For now, we'll create a placeholder that shows processed image characteristics
        
        sample_datasets = ['aptos2019', 'idrid_grading']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Processed Image Quality Assessment', fontsize=16, fontweight='bold')
        
        for dataset_idx, dataset_name in enumerate(sample_datasets):
            # Get sample images
            samples = self.sample_images_from_split(dataset_name, 'train', 2)
            
            if dataset_name == 'aptos2019':
                images_dir = self.processed_dir / "aptos2019" / "images"
            else:
                images_dir = self.processed_dir / "grading" / "images"
            
            for img_idx, img_id in enumerate(samples):
                ax = axes[dataset_idx, img_idx * 2]
                ax_hist = axes[dataset_idx, img_idx * 2 + 1]
                
                img_path = images_dir / f"{img_id}.jpg"
                
                if img_path.exists():
                    try:
                        img = cv2.imread(str(img_path))
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Display image
                        ax.imshow(img_rgb)
                        ax.set_title(f'{dataset_name}\\n{img_id}')
                        ax.axis('off')
                        
                        # Display histogram
                        colors = ['red', 'green', 'blue']
                        for i, color in enumerate(colors):
                            ax_hist.hist(img_rgb[:,:,i].flatten(), bins=50, alpha=0.6, 
                                       label=color, color=color, density=True)
                        
                        ax_hist.set_title(f'RGB Distribution\\n{img_id}')
                        ax_hist.set_xlabel('Pixel Value')
                        ax_hist.set_ylabel('Density')
                        ax_hist.legend()
                        ax_hist.grid(True, alpha=0.3)
                        
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                        ax_hist.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                    ax_hist.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        plt.tight_layout()
        save_path = self.visual_check_dir / "preprocessing_quality_check.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved preprocessing quality check to {save_path}")
    
    def run_all_visual_checks(self) -> None:
        """Run all visual quality checks."""
        print("="*60)
        print("RUNNING COMPREHENSIVE VISUAL QUALITY CHECKS")
        print("="*60)
        
        # Set random seed for reproducible sampling
        random.seed(42)
        
        # Classification spot-checks
        self.create_classification_spot_check('aptos2019')
        self.create_classification_spot_check('idrid_grading')
        
        # Segmentation spot-check
        self.create_segmentation_spot_check('idrid_segmentation')
        
        # Preprocessing quality check
        self.create_preprocessing_comparison()
        
        print(f"\\nAll visual checks completed!")
        print(f"Results saved to: {self.visual_check_dir}")
        print("\\nReview the generated images to verify:")
        print("- Image quality and consistency")
        print("- Proper image-mask alignment (segmentation)")
        print("- No obvious corruption or artifacts")
        print("- Consistent preprocessing across samples")

def main():
    """Run visual quality checks."""
    checker = VisualQualityChecker()
    checker.run_all_visual_checks()

if __name__ == "__main__":
    main()
