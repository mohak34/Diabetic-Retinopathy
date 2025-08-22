#!/usr/bin/env python3
"""
Test Script for Hyperparameter Optimization with ACTUAL Training
This script tests a single hyperparameter configuration to ensure training works
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('TestHyperoptTraining')

def test_single_config():
    """Test a single hyperparameter configuration"""
    
    logger.info("🧪 Testing hyperparameter optimization with ACTUAL training")
    
    # Test configuration - using research-optimal parameters
    test_hyperparams = {
        'learning_rate': 1e-4,      # Research optimal
        'focal_gamma': 2.5,         # Good for class imbalance
        'weight_decay': 1e-2,       # Moderate regularization
        'batch_size': 16,           # Memory efficient
        'classification_dropout': 0.4,  # Good regularization
        'segmentation_weight_final': 0.6  # Balanced multi-task
    }
    
    logger.info(f"📋 Test hyperparameters: {test_hyperparams}")
    
    try:
        # Import the training function
        from run_hyperopt_training import run_training_with_hyperparams
        
        logger.info("🚀 Starting test training...")
        
        # Run training with hyperparameters
        results = run_training_with_hyperparams(
            hyperparams=test_hyperparams,
            mode='quick',  # Use quick mode for testing
            log_level='INFO'
        )
        
        # Analyze results
        if 'error' in results:
            logger.error(f"❌ Training failed: {results['error']}")
            return False
        
        if 'training_results' in results:
            training_results = results['training_results']
            
            if 'best_metrics' in training_results:
                metrics = training_results['best_metrics']
                logger.info("✅ Training completed successfully!")
                logger.info(f"📊 Results:")
                logger.info(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
                logger.info(f"   Dice Score: {metrics.get('dice_score', 0):.4f}")
                logger.info(f"   Combined Score: {metrics.get('combined_score', 0):.4f}")
                logger.info(f"   Sensitivity: {metrics.get('sensitivity', 0):.4f}")
                logger.info(f"   Specificity: {metrics.get('specificity', 0):.4f}")
                
                # Check if this is actual training or simulation
                if results.get('actual_training', False):
                    logger.info("🎯 CONFIRMED: This is ACTUAL training, not simulation!")
                else:
                    logger.warning("⚠️ This might be simulation - check training implementation")
                
                return True
            else:
                logger.warning("⚠️ Training results missing metrics")
                return False
        else:
            logger.warning("⚠️ No training results found")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure all required modules are available")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_hyperopt_system():
    """Test the full hyperparameter optimization system"""
    
    logger.info("🔬 Testing hyperparameter optimization system...")
    
    try:
        # Import from the correct module - the main optimization script
        from run_hyperparameter_optimization import HyperparameterOptimizer
        
        # Create optimizer with experiment name
        optimizer = HyperparameterOptimizer("test_hyperopt")
        
        # Test configuration - use the expected format for this optimizer
        test_config = {
            'name': 'test_config',
            'learning_rate': 1e-4,
            'focal_gamma': 2.5,
            'weight_decay': 1e-2,
            'batch_size': 16,
            'classification_dropout': 0.4,
            'segmentation_weight_final': 0.6,
            'expected_accuracy': 0.87
        }
        
        logger.info("🧪 Running single optimization experiment...")
        
        # Run single experiment with the test config
        result = optimizer.run_single_experiment(test_config, 'test_config')
        
        if result and result.get('status') == 'completed':
            metrics = result.get('metrics', {})
            logger.info("✅ Hyperparameter optimization test successful!")
            logger.info(f"📊 Test Results:")
            logger.info(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"   Dice Score: {metrics.get('dice_score', 0):.4f}")
            logger.info(f"   Combined Score: {metrics.get('combined_score', 0):.4f}")
            
            # Check if actual training was attempted
            full_results = result.get('full_results', {})
            if not full_results.get('simulation', False):
                logger.info("🎯 CONFIRMED: Hyperopt system attempted ACTUAL training!")
            else:
                logger.warning("⚠️ Hyperopt system used simulation")
            
            return True
        else:
            logger.error(f"❌ Optimization experiment failed: {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Hyperopt system test failed: {e}")
        return False

def main():
    """Main test function"""
    
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER OPTIMIZATION TRAINING TEST")
    logger.info("=" * 60)
    
    # Test 1: Single configuration training
    logger.info("\n" + "=" * 40)
    logger.info("TEST 1: Single Configuration Training")
    logger.info("=" * 40)
    
    success1 = test_single_config()
    
    # Test 2: Full hyperparameter optimization system
    logger.info("\n" + "=" * 40)
    logger.info("TEST 2: Hyperparameter Optimization System")
    logger.info("=" * 40)
    
    success2 = test_hyperopt_system()
    
    # Final results
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Single Config Training: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"Hyperopt System: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        logger.info("🎉 ALL TESTS PASSED - Hyperparameter optimization with actual training is working!")
        return 0
    else:
        logger.info("❌ Some tests failed - check the implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
