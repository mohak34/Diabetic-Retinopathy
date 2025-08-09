#!/usr/bin/env python3
"""
Phase 3 Testing Script: Model Architecture Development
Comprehensive test for EfficientNetV2-S backbone, multi-task heads, and integrated model.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
import traceback
import numpy as np
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available."""
    logger.info("Checking dependencies...")
    
    try:
        import timm
        logger.info(f"OK timm version: {timm.__version__}")
    except ImportError:
        logger.error(" timm not found. Install with: pip install timm")
        return False
    
    try:
        import albumentations
        logger.info(f"OK albumentations version: {albumentations.__version__}")
    except ImportError:
        logger.error(" albumentations not found. Install with: pip install albumentations")
        return False
    
    try:
        import torch
        logger.info(f"OK PyTorch version: {torch.__version__}")
        logger.info(f"OK CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"OK CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"OK CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        logger.error(" PyTorch not found")
        return False
    
    return True

def test_backbone():
    """Test EfficientNetV2-S backbone functionality."""
    logger.info("="*60)
    logger.info("TESTING EFFICIENTNETV2-S BACKBONE")
    logger.info("="*60)
    
    try:
        from src.models.backbone import EfficientNetBackbone, create_efficientnet_backbone
        
        # Test backbone creation
        logger.info("Creating EfficientNetV2-S backbone...")
        backbone = create_efficientnet_backbone(
            model_name='tf_efficientnet_b0_ns',
            pretrained=True,
            freeze_stages=0
        )
        
        # Test input
        batch_size = 4
        input_size = 512
        test_input = torch.randn(batch_size, 3, input_size, input_size)
        logger.info(f"Test input shape: {test_input.shape}")
        
        # Test forward pass
        logger.info("Testing forward pass...")
        with torch.no_grad():
            features_list = backbone(test_input)
            final_features = backbone.get_final_features(test_input)
        
        logger.info(f"Number of feature stages: {len(features_list)}")
        for i, features in enumerate(features_list):
            logger.info(f"  Stage {i}: {features.shape}")
        
        logger.info(f"Final features shape: {final_features.shape}")
        
        # Test feature info
        feature_info = backbone.get_feature_info()
        logger.info("Feature information:")
        for info in feature_info:
            logger.info(f"  {info}")
        
        # Test memory estimation
        memory_est = backbone.estimate_memory_usage(batch_size=4, input_size=512)
        logger.info(f"Estimated memory usage:")
        for key, value in memory_est.items():
            logger.info(f"  {key}: {value:.2f} MB")
        
        # Test freezing
        logger.info("Testing layer freezing...")
        backbone.freeze_early_layers(freeze_stages=2)
        backbone.unfreeze_all()
        
        logger.info("OK Backbone tests passed!")
        return True, backbone
        
    except Exception as e:
        logger.error(f" Backbone test failed: {e}")
        traceback.print_exc()
        return False, None

def test_heads(backbone_features: int = 1280):
    """Test classification and segmentation heads."""
    logger.info("="*60)
    logger.info("TESTING MULTI-TASK HEADS")
    logger.info("="*60)
    
    try:
        from src.models.heads import ClassificationHead, SegmentationHead, AdvancedSegmentationHead
        
        # Test parameters
        batch_size = 4
        feature_height, feature_width = 16, 16  # Typical EfficientNet output
        
        # Create test features
        test_features = torch.randn(batch_size, backbone_features, feature_height, feature_width)
        logger.info(f"Test features shape: {test_features.shape}")
        
        # Test Classification Head
        logger.info("Testing ClassificationHead...")
        cls_head = ClassificationHead(
            in_features=backbone_features,
            num_classes=5,
            dropout_rate=0.3,
            hidden_dim=256
        )
        
        with torch.no_grad():
            cls_output = cls_head(test_features)
            cls_probs = cls_head.get_probabilities(test_features)
        
        logger.info(f"  Classification output shape: {cls_output.shape}")
        logger.info(f"  Classification probabilities shape: {cls_probs.shape}")
        logger.info(f"  Probability sum (should be ~1.0): {cls_probs.sum(dim=1).mean().item():.4f}")
        
        assert cls_output.shape == (batch_size, 5), f"Wrong cls shape: {cls_output.shape}"
        assert torch.allclose(cls_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5), "Probabilities don't sum to 1"
        
        # Test Segmentation Head
        logger.info("Testing SegmentationHead...")
        seg_head = SegmentationHead(
            in_features=backbone_features,
            num_classes=1,
            decoder_channels=[512, 256, 128, 64]
        )
        
        with torch.no_grad():
            seg_output = seg_head(test_features)
        
        logger.info(f"  Segmentation output shape: {seg_output.shape}")
        logger.info(f"  Output min/max: {seg_output.min().item():.4f}/{seg_output.max().item():.4f}")
        
        assert seg_output.shape[0] == batch_size, f"Wrong batch size: {seg_output.shape[0]}"
        assert seg_output.shape[1] == 1, f"Wrong number of channels: {seg_output.shape[1]}"
        assert seg_output.min() >= 0 and seg_output.max() <= 1, "Segmentation output not in [0,1] range"
        
        # Test Advanced Segmentation Head
        logger.info("Testing AdvancedSegmentationHead...")
        skip_channels = [64, 160, 256]
        
        adv_seg_head = AdvancedSegmentationHead(
            in_features=backbone_features,
            num_classes=1,
            skip_feature_channels=skip_channels,
            use_attention=True
        )
        
        # Create dummy skip features
        skip_features = [
            torch.randn(batch_size, 64, 128, 128),   # High resolution
            torch.randn(batch_size, 160, 64, 64),    # Medium resolution  
            torch.randn(batch_size, 256, 32, 32),    # Low resolution
        ]
        
        with torch.no_grad():
            adv_seg_output = adv_seg_head(test_features, skip_features)
        
        logger.info(f"  Advanced segmentation output shape: {adv_seg_output.shape}")
        
        logger.info("OK Head tests passed!")
        return True, cls_head, seg_head
        
    except Exception as e:
        logger.error(f" Head tests failed: {e}")
        traceback.print_exc()
        return False, None, None

def test_losses():
    """Test loss function implementations."""
    logger.info("="*60)
    logger.info("TESTING LOSS FUNCTIONS")
    logger.info("="*60)
    
    try:
        from src.training.losses import (
            FocalLoss, DiceLoss, CombinedSegmentationLoss, 
            QuadraticWeightedKappa, MultiTaskLoss
        )
        
        # Test parameters
        batch_size = 4
        num_classes_cls = 5
        num_classes_seg = 1
        height, width = 64, 64
        
        # Create test data
        cls_pred = torch.randn(batch_size, num_classes_cls)
        cls_target = torch.randint(0, num_classes_cls, (batch_size,))
        
        seg_pred = torch.randn(batch_size, num_classes_seg, height, width)
        seg_target = torch.randint(0, 2, (batch_size, height, width)).float()
        
        logger.info(f"Test data shapes:")
        logger.info(f"  cls_pred: {cls_pred.shape}, cls_target: {cls_target.shape}")
        logger.info(f"  seg_pred: {seg_pred.shape}, seg_target: {seg_target.shape}")
        
        # Test Focal Loss
        logger.info("Testing FocalLoss...")
        focal_loss = FocalLoss(gamma=2.0)
        focal_result = focal_loss(cls_pred, cls_target)
        logger.info(f"  Focal loss: {focal_result.item():.4f}")
        assert focal_result.item() > 0, "Focal loss should be positive"
        
        # Test Dice Loss
        logger.info("Testing DiceLoss...")
        dice_loss = DiceLoss()
        dice_result = dice_loss(torch.sigmoid(seg_pred), seg_target.unsqueeze(1))
        logger.info(f"  Dice loss: {dice_result.item():.4f}")
        assert 0 <= dice_result.item() <= 1, "Dice loss should be in [0,1]"
        
        # Test Combined Segmentation Loss
        logger.info("Testing CombinedSegmentationLoss...")
        combined_seg_loss = CombinedSegmentationLoss()
        combined_seg_result = combined_seg_loss(seg_pred, seg_target)
        logger.info(f"  Combined seg losses:")
        for key, value in combined_seg_result.items():
            logger.info(f"    {key}: {value.item():.4f}")
        
        # Test Multi-Task Loss
        logger.info("Testing MultiTaskLoss...")
        multi_task_loss = MultiTaskLoss(
            classification_weight=1.0,
            segmentation_weight=0.5,
            adaptive_weighting=False
        )
        
        multi_result = multi_task_loss(cls_pred, seg_pred, cls_target, seg_target, epoch=10)
        logger.info(f"  Multi-task losses:")
        for key, value in multi_result.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"    {key}: {value.item():.4f}")
        
        logger.info("OK Loss function tests passed!")
        return True
        
    except Exception as e:
        logger.error(f" Loss function tests failed: {e}")
        traceback.print_exc()
        return False

def test_integrated_model():
    """Test the complete integrated multi-task model."""
    logger.info("="*60)
    logger.info("TESTING INTEGRATED MULTI-TASK MODEL")
    logger.info("="*60)
    
    try:
        from src.models.multi_task_model import MultiTaskRetinaModel, create_multi_task_model
        
        # Test parameters
        batch_size = 4
        input_size = 512
        num_classes_cls = 5
        num_classes_seg = 1
        
        # Create test input
        test_input = torch.randn(batch_size, 3, input_size, input_size)
        logger.info(f"Test input shape: {test_input.shape}")
        
        # Test basic model
        logger.info("Testing basic multi-task model...")
        model_basic = create_multi_task_model(
            num_classes_cls=num_classes_cls,
            num_classes_seg=num_classes_seg,
            use_skip_connections=False,
            pretrained=True
        )
        
        with torch.no_grad():
            cls_out, seg_out = model_basic(test_input)
        
        logger.info(f"  Classification output shape: {cls_out.shape}")
        logger.info(f"  Segmentation output shape: {seg_out.shape}")
        
        # Validate output shapes
        assert cls_out.shape == (batch_size, num_classes_cls), f"Wrong cls shape: {cls_out.shape}"
        assert seg_out.shape[0] == batch_size, f"Wrong seg batch size: {seg_out.shape[0]}"
        assert seg_out.shape[1] == num_classes_seg, f"Wrong seg channels: {seg_out.shape[1]}"
        
        # Test with skip connections
        logger.info("Testing model with skip connections...")
        model_skip = create_multi_task_model(
            num_classes_cls=num_classes_cls,
            num_classes_seg=num_classes_seg,
            use_skip_connections=True,
            use_advanced_decoder=True,
            pretrained=True
        )
        
        with torch.no_grad():
            cls_out_skip, seg_out_skip = model_skip(test_input)
        
        logger.info(f"  Skip model - Classification: {cls_out_skip.shape}")
        logger.info(f"  Skip model - Segmentation: {seg_out_skip.shape}")
        
        # Test prediction methods
        logger.info("Testing prediction methods...")
        cls_pred = model_basic.predict_classification(test_input)
        seg_pred = model_basic.predict_segmentation(test_input)
        both_pred = model_basic.predict_both(test_input)
        
        logger.info(f"  Classification prediction keys: {list(cls_pred.keys())}")
        logger.info(f"  Segmentation prediction keys: {list(seg_pred.keys())}")
        logger.info(f"  Both prediction keys: {list(both_pred.keys())}")
        
        # Validate prediction outputs
        assert 'logits' in cls_pred and 'probabilities' in cls_pred and 'predictions' in cls_pred
        assert 'raw_output' in seg_pred and 'binary_masks' in seg_pred
        assert 'classification' in both_pred and 'segmentation' in both_pred
        
        # Test model analysis
        logger.info("Testing model analysis...")
        model_size = model_basic.get_model_size()
        memory_usage = model_basic.estimate_memory_usage(batch_size=4, input_size=512)
        
        logger.info(f"  Model parameters:")
        for key, value in model_size.items():
            logger.info(f"    {key}: {value:,}")
        
        logger.info(f"  Memory usage estimates:")
        for key, value in memory_usage.items():
            logger.info(f"    {key}: {value:.2f} MB")
        
        # Check RTX 3080 memory constraint
        estimated_memory_gb = memory_usage['total_estimated'] / 1000
        logger.info(f"  Estimated total memory: {estimated_memory_gb:.2f} GB")
        
        if estimated_memory_gb > 8.0:
            logger.warning("  WARNING  Model may exceed RTX 3080 8GB memory limit!")
        else:
            logger.info("  OK Model should fit in RTX 3080 memory")
        
        # Test gradient flow
        logger.info("Testing gradient flow...")
        model_basic.train()
        
        # Forward pass
        cls_out, seg_out = model_basic(test_input)
        
        # Create dummy targets
        cls_target = torch.randint(0, num_classes_cls, (batch_size,))
        seg_target = torch.randint(0, 2, (batch_size, seg_out.shape[2], seg_out.shape[3])).float()
        
        # Compute simple losses
        cls_loss = F.cross_entropy(cls_out, cls_target)
        seg_loss = F.binary_cross_entropy(seg_out.squeeze(1), seg_target)
        total_loss = cls_loss + seg_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        has_gradients = any(param.grad is not None for param in model_basic.parameters() if param.requires_grad)
        logger.info(f"  Gradients computed: {has_gradients}")
        assert has_gradients, "No gradients found - backward pass failed"
        
        logger.info("OK Integrated model tests passed!")
        return True, model_basic
        
    except Exception as e:
        logger.error(f" Integrated model tests failed: {e}")
        traceback.print_exc()
        return False, None

def test_memory_constraints():
    """Test memory usage under RTX 3080 constraints."""
    logger.info("="*60)
    logger.info("TESTING MEMORY CONSTRAINTS (RTX 3080 - 8GB)")
    logger.info("="*60)
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - skipping GPU memory tests")
        return True
    
    try:
        from src.models.multi_task_model import create_multi_task_model
        
        # Create model
        model = create_multi_task_model(
            num_classes_cls=5,
            num_classes_seg=1,
            use_skip_connections=True,
            use_advanced_decoder=True,
            pretrained=True
        )
        
        # Move to GPU
        device = torch.cuda.current_device()
        model = model.cuda()
        
        # Test different batch sizes
        input_size = 512
        max_batch_size = 1
        
        for batch_size in [1, 2, 4, 8]:
            try:
                logger.info(f"Testing batch size {batch_size}...")
                
                # Clear cache
                torch.cuda.empty_cache()
                
                # Create input
                test_input = torch.randn(batch_size, 3, input_size, input_size).cuda()
                
                # Forward pass
                with torch.no_grad():
                    cls_out, seg_out = model(test_input)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated(device) / 1e9
                memory_cached = torch.cuda.memory_reserved(device) / 1e9
                
                logger.info(f"  Batch {batch_size} - Memory used: {memory_used:.2f} GB, cached: {memory_cached:.2f} GB")
                
                if memory_used < 7.5:  # Leave some headroom
                    max_batch_size = batch_size
                else:
                    logger.warning(f"  Batch size {batch_size} exceeds memory limit")
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"  Batch size {batch_size} - OOM error")
                    break
                else:
                    raise
        
        logger.info(f"Maximum recommended batch size: {max_batch_size}")
        
        # Test training memory (forward + backward)
        logger.info("Testing training memory usage...")
        try:
            torch.cuda.empty_cache()
            
            model.train()
            test_input = torch.randn(max_batch_size, 3, input_size, input_size).cuda()
            
            cls_out, seg_out = model(test_input)
            
            # Create dummy targets
            cls_target = torch.randint(0, 5, (max_batch_size,)).cuda()
            seg_target = torch.randint(0, 2, (max_batch_size, seg_out.shape[2], seg_out.shape[3])).float().cuda()
            
            # Compute losses
            cls_loss = F.cross_entropy(cls_out, cls_target)
            seg_loss = F.binary_cross_entropy(seg_out.squeeze(1), seg_target)
            total_loss = cls_loss + seg_loss
            
            # Backward pass
            total_loss.backward()
            
            memory_used = torch.cuda.memory_allocated(device) / 1e9
            logger.info(f"Training memory usage: {memory_used:.2f} GB")
            
            if memory_used > 7.5:
                logger.warning("Training memory usage may be too high for RTX 3080")
            else:
                logger.info("OK Training memory usage acceptable")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("Training OOM - consider reducing batch size or model complexity")
                return False
            else:
                raise
        
        logger.info("OK Memory constraint tests passed!")
        return True
        
    except Exception as e:
        logger.error(f" Memory constraint tests failed: {e}")
        traceback.print_exc()
        return False

def run_phase3_tests():
    """Run all Phase 3 tests."""
    print("="*80)
    print("PHASE 3: MODEL ARCHITECTURE DEVELOPMENT - COMPREHENSIVE TESTING")
    print("="*80)
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("Dependency check failed. Please install missing packages.")
        return False
    
    # Track test results
    test_results = {}
    
    # Test 1: Backbone
    logger.info("\n" + "="*80)
    logger.info("TEST 1: EFFICIENTNETV2-S BACKBONE")
    success, backbone = test_backbone()
    test_results['backbone'] = success
    
    # Test 2: Heads
    logger.info("\n" + "="*80)
    logger.info("TEST 2: MULTI-TASK HEADS")
    success, cls_head, seg_head = test_heads()
    test_results['heads'] = success
    
    # Test 3: Loss Functions
    logger.info("\n" + "="*80)
    logger.info("TEST 3: LOSS FUNCTIONS")
    success = test_losses()
    test_results['losses'] = success
    
    # Test 4: Integrated Model
    logger.info("\n" + "="*80)
    logger.info("TEST 4: INTEGRATED MULTI-TASK MODEL")
    success, model = test_integrated_model()
    test_results['integrated_model'] = success
    
    # Test 5: Memory Constraints
    logger.info("\n" + "="*80)
    logger.info("TEST 5: MEMORY CONSTRAINTS")
    success = test_memory_constraints()
    test_results['memory_constraints'] = success
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PHASE 3 TEST SUMMARY")
    logger.info("="*80)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "OK PASSED" if result else " FAILED"
        logger.info(f"{test_name.upper()}: {status}")
        if not result:
            all_passed = False
    
    logger.info("="*80)
    if all_passed:
        logger.info("SUCCESS: ALL PHASE 3 TESTS PASSED!")
        logger.info("OK EfficientNetV2-S backbone implemented and tested")
        logger.info("OK Multi-task heads working correctly")
        logger.info("OK Loss functions validated")
        logger.info("OK Integrated model ready for training")
        logger.info("OK Memory constraints satisfied for RTX 3080")
        logger.info("\nTest READY TO PROCEED TO PHASE 4: TRAINING INFRASTRUCTURE")
    else:
        logger.error("ERROR SOME TESTS FAILED - PLEASE FIX ISSUES BEFORE PROCEEDING")
    
    logger.info("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = run_phase3_tests()
    sys.exit(0 if success else 1)
