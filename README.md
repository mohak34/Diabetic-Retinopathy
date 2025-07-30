# Diabetic Retinopathy Detection System

## Project Overview

This project implements an end-to-end deep learning system for automated diabetic retinopathy detection and lesion segmentation using retinal fundus images. The system processes multiple datasets (APTOS 2019, IDRiD) and provides both severity grading and localization capabilities.

## Implementation Progress

### Phase 1: Data Processing and Preparation

**Objective**: Process raw medical imaging datasets into a standardized format suitable for deep learning workflows.

**What Was Implemented**:

- Automated download and extraction of APTOS 2019 Diabetic Retinopathy dataset
- Processing of IDRiD (Indian Diabetic Retinopathy Image Dataset) for both grading and segmentation tasks
- Image standardization to 512x512 resolution with quality preservation
- Metadata extraction and validation for all datasets
- Organized directory structure for processed data

**Key Components**:

- `main.py`: Main data processing pipeline orchestrator
- `src/data_processing/`: Core processing modules for each dataset
- `src/utils/`: Utility functions for image operations and file management

**Data Statistics After Processing**:

- APTOS 2019: 3,662 fundus images with severity grades (0-4)
- IDRiD Grading: 516 images with detailed severity annotations
- IDRiD Segmentation: 363 image-mask pairs for lesion localization

**How to Run Phase 1**:

```bash
uv run python main.py
```

### Phase 2: Data Pipeline Implementation

#### Phase 2.1: Data Analysis and Exploration

**Objective**: Comprehensive analysis of processed datasets to understand distribution patterns and data quality.

**What Was Implemented**:

- Statistical analysis of image dimensions, file sizes, and quality metrics
- Class distribution analysis for severity grading datasets
- Correlation analysis between different severity grades
- Data quality assessment and outlier detection

**Key Files**:

- `count_images.py`: Dataset statistics and validation
- Analysis notebooks in `notebooks/` directory

#### Phase 2.2: Data Augmentation and Dataset Classes

**Objective**: Create robust data augmentation pipelines and PyTorch dataset classes optimized for medical imaging.

**What Was Implemented**:

1. **Data Augmentation Pipelines** (`src/data/transforms.py`):

   - Medical imaging-specific augmentations using Albumentations library
   - Classification transforms: 16 operations including geometric, photometric, and noise augmentations
   - Segmentation transforms: Coordinated image-mask augmentations preserving spatial relationships
   - Validation transforms: Minimal preprocessing for consistent evaluation
   - Test-time augmentation strategies for improved inference accuracy

2. **Custom PyTorch Dataset Classes** (`src/data/datasets.py`):

   - `GradingRetinaDataset`: Handles diabetic retinopathy severity classification tasks
   - `SegmentationRetinaDataset`: Manages lesion segmentation with image-mask pairs
   - `MultiTaskRetinaDataset`: Supports combined grading and segmentation learning
   - Robust error handling for missing files and corrupted data
   - Flexible transform integration with runtime augmentation pipeline support

3. **Stratified Data Splits** (`src/data/create_simple_splits.py`):

   - APTOS 2019: 2,929 training / 733 validation images (80/20 stratified split)
   - IDRiD Grading: 412 training / 104 validation images (80/20 stratified split)
   - IDRiD Segmentation: 291 training / 72 validation image-mask pairs (80/20 stratified split)
   - Class-balanced stratification preserving original label distributions
   - Reproducible splits using fixed random state (42)

4. **Optimized DataLoader Factory** (`src/data/dataloaders.py`):
   - RTX 3080 Mobile optimized configuration (batch_size=4, num_workers=2)
   - Memory management with pin_memory=True and persistent_workers=True
   - Weighted sampling for handling class imbalance automatically
   - Flexible parameter configuration for different GPU configurations
   - Multi-task learning support for combined datasets

**Key Features**:

- Medical image preservation: Augmentations designed to maintain diagnostic characteristics
- GPU memory optimization: Configured for RTX 3080 Mobile constraints
- Reproducible experiments: Fixed random seeds and consistent data splits
- Multi-task capability: Support for simultaneous grading and segmentation learning

**Validation Results**:

- All module imports successful
- Transform pipelines functional (16 training operations, 4 validation operations)
- Data splits created and validated for all three datasets
- DataLoader factory optimized and operational
- End-to-end pipeline integration verified

**How to Run Phase 2.2 Components**:

1. Create data splits:

```bash
uv run python src/data/create_simple_splits.py
```

2. Test complete pipeline:

```bash
uv run python src/data/test_pipeline.py
```

3. Quick component validation:

```bash
uv run python quick_test.py
```

## Current Data Pipeline Architecture

### Directory Structure

```
data/
├── processed/
│   ├── aptos2019/          # 3,662 processed fundus images
│   ├── grading/            # 516 IDRiD grading images
│   ├── segmentation/       # 363 IDRiD segmentation pairs
│   └── localization/       # (Future: lesion localization data)
├── splits/
│   ├── aptos2019_splits.json
│   ├── idrid_grading_splits.json
│   └── idrid_segmentation_splits.json
└── metadata/
    ├── aptos2019_metadata.json
    ├── idrid_grading_metadata.json
    └── idrid_segmentation_metadata.json
```

### Code Organization

```
src/
├── data/
│   ├── transforms.py       # Augmentation pipelines
│   ├── datasets.py         # PyTorch Dataset classes
│   ├── dataloaders.py      # DataLoader factory
│   ├── create_simple_splits.py  # Data splitting utility
│   └── test_pipeline.py    # Pipeline validation
├── data_processing/
│   ├── aptos_processor.py  # APTOS dataset processing
│   ├── idrid_processor.py  # IDRiD dataset processing
│   └── base_processor.py   # Common processing utilities
└── utils/
    ├── image_utils.py      # Image processing utilities
    ├── file_utils.py       # File management utilities
    └── logging_utils.py    # Logging configuration
```

## Technical Specifications

### Data Augmentation Strategy

- **Geometric**: Rotation (15°), scaling (±10%), translation
- **Photometric**: Brightness/contrast adjustment, gamma correction
- **Quality**: CLAHE (adaptive histogram equalization)
- **Noise**: Gaussian noise, compression artifacts simulation
- **Occlusion**: Coarse dropout for robustness testing
- **Medical-specific**: Preserves vessel structure and lesion characteristics

### Dataset Split Strategy

- **Method**: Stratified random sampling maintaining class distribution
- **Ratio**: 80% training, 20% validation
- **Validation**: Class proportions preserved across splits
- **Reproducibility**: Fixed random state (42) for consistent results

### Hardware Optimization

- **Target GPU**: RTX 3080 Mobile (8GB VRAM)
- **Batch Size**: 4 (optimal for memory constraints)
- **Workers**: 2 (CPU core utilization)
- **Memory**: Pin memory enabled, persistent workers for efficiency

## Dependencies

Core dependencies managed through `pyproject.toml`:

- PyTorch: Deep learning framework
- Albumentations: Advanced image augmentation
- OpenCV: Computer vision operations
- Pillow: Image processing
- NumPy: Numerical computations
- Scikit-learn: Data splitting and metrics
- Pandas: Data manipulation

## Next Phase: Model Implementation

Phase 2.3 will implement:

- EfficientNetV2-S backbone architecture
- Multi-task learning heads for grading and segmentation
- Transfer learning from ImageNet pretrained weights
- Model compilation and optimization for training

The data pipeline is now complete and ready for model training with robust augmentation, balanced sampling, and optimized loading for the target hardware configuration.

## Setup and Installation

This project uses `uv` for dependency management. The project includes a `uv.lock` file that ensures reproducible builds.

1. **Install `uv`**: If you don't have `uv` installed, you can install it using pip:

   ```bash
   pip install uv
   ```

2. **Run the project**: Navigate to the project directory and use `uv run` to automatically install dependencies and run scripts:

   ```bash
   uv run python main.py
   ```

The `uv run` command will automatically create a virtual environment and install all dependencies from the lock file on first use.
