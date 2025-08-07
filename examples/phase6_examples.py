"""
Phase 6 Usage Example
Demonstrates how to use the Phase 6 evaluation pipeline for comprehensive model evaluation.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level to project root
sys.path.append(str(project_root))

# Import the Phase6EvaluationPipeline from scripts
sys.path.insert(0, str(project_root / 'scripts'))
from phase6_evaluation_main import Phase6EvaluationPipeline


def example_phase6_evaluation():
    """
    Example usage of Phase 6 evaluation pipeline
    """
    
    print("=" * 80)
    print("PHASE 6: MODEL EVALUATION & ANALYSIS - USAGE EXAMPLE")
    print("=" * 80)
    
    # Configuration
    config_path = "configs/phase6_evaluation_config.yaml"
    output_dir = "results/phase6_evaluation_example"
    
    # Check for existing model checkpoints or create test checkpoint
    checkpoint_paths = []
    
    # Look for existing checkpoints in experiments
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                checkpoint_dir = exp_dir / "checkpoints"
                if checkpoint_dir.exists():
                    for checkpoint_file in checkpoint_dir.glob("*.pth"):
                        checkpoint_paths.append(str(checkpoint_file))
    
    # If no checkpoints found, create a test checkpoint
    if not checkpoint_paths:
        print("No existing checkpoints found. Creating test checkpoint...")
        try:
            # Create test checkpoint
            import sys
            from pathlib import Path
            tests_dir = Path(__file__).parent.parent / 'tests'
            sys.path.insert(0, str(tests_dir))
            from create_test_checkpoint import create_dummy_checkpoint
            test_checkpoint = create_dummy_checkpoint()
            checkpoint_paths = [test_checkpoint]
            print(f"Created test checkpoint: {test_checkpoint}")
        except Exception as e:
            print(f"Failed to create test checkpoint: {e}")
            print("Please ensure you have trained models available or check the model implementation.")
            return
    else:
        # Use only the first few checkpoints for demo
        checkpoint_paths = checkpoint_paths[:2]
    
    print(f"Using checkpoints: {checkpoint_paths}")
    
    # Initialize Phase 6 pipeline
    print(f"\n1. Initializing Phase 6 evaluation pipeline...")
    print(f"   Config: {config_path}")
    print(f"   Output directory: {output_dir}")
    print(f"   Checkpoints to evaluate: {len(checkpoint_paths)}")
    
    try:
        pipeline = Phase6EvaluationPipeline(
            config_path=config_path,
            output_dir=output_dir
        )
        
        print("   ✓ Pipeline initialized successfully")
        
    except Exception as e:
        print(f"   ✗ Failed to initialize pipeline: {e}")
        return
    
    # Run comprehensive evaluation
    print(f"\n2. Running comprehensive Phase 6 evaluation...")
    
    try:
        results = pipeline.run_complete_evaluation(
            checkpoint_paths=checkpoint_paths,
            datasets_config=None  # Use default from config
        )
        
        print("   ✓ Evaluation completed successfully")
        
        # Print summary of results
        print_results_summary(results)
        
    except Exception as e:
        print(f"   ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n" + "=" * 80)
    print("PHASE 6 EVALUATION EXAMPLE COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Final report: {results.get('final_report_path', 'Not generated')}")


def print_results_summary(results):
    """Print a summary of evaluation results"""
    
    print(f"\n3. Results Summary:")
    
    # Count evaluations
    checkpoint_results = {k: v for k, v in results.items() 
                         if k not in ['comparative_analysis', 'final_report_path']}
    
    print(f"   Checkpoints evaluated: {len(checkpoint_results)}")
    
    # Summarize performance for each checkpoint
    for checkpoint_name, checkpoint_result in checkpoint_results.items():
        print(f"\n   📊 Checkpoint: {checkpoint_name}")
        
        # Internal evaluation summary
        if 'internal_evaluation' in checkpoint_result:
            internal_results = checkpoint_result['internal_evaluation']
            print(f"      Internal datasets evaluated: {len(internal_results)}")
            
            # Get best internal performance
            if internal_results:
                best_score = max([r.combined_score for r in internal_results.values()])
                print(f"      Best internal combined score: {best_score:.4f}")
        
        # External validation summary
        if 'external_validation' in checkpoint_result:
            external_validation = checkpoint_result['external_validation']
            if 'external_results' in external_validation:
                external_results = external_validation['external_results']
                print(f"      External datasets evaluated: {len(external_results)}")
                
                if external_results:
                    best_external_score = max([r.combined_score for r in external_results.values()])
                    print(f"      Best external combined score: {best_external_score:.4f}")
            
            # Generalization analysis summary
            if 'generalization_analysis' in external_validation:
                gen_analysis = external_validation['generalization_analysis']
                if 'overall_analysis' in gen_analysis:
                    overall = gen_analysis['overall_analysis']
                    print(f"      Generalization score: {overall['generalization_score']:.4f}")
                    print(f"      Risk level: {overall['risk_assessment']['overall_risk']}")
        
        # Explainability summary
        if 'explainability_analysis' in checkpoint_result:
            explainability = checkpoint_result['explainability_analysis']
            if 'insights' in explainability:
                insights = explainability['insights']
                print(f"      Explainability insights generated: ✓")
                
                if 'grade_separation' in insights:
                    sep_analysis = insights['grade_separation']
                    most_sep = sep_analysis['most_separable']
                    print(f"      Most separable grades: {most_sep[0]} (distance: {most_sep[1]:.3f})")
    
    # Comparative analysis summary
    if 'comparative_analysis' in results:
        print(f"\n   🔄 Comparative Analysis:")
        comp_analysis = results['comparative_analysis']
        
        if 'best_performing_models' in comp_analysis:
            best_models = comp_analysis['best_performing_models']
            print(f"      Best models identified for {len(best_models)} metrics")
            
            # Show best overall model
            if 'combined_score' in best_models:
                best_overall = best_models['combined_score']
                print(f"      Best overall: {best_overall['checkpoint']} (score: {best_overall['score']:.4f})")


def example_step_by_step_evaluation():
    """
    Example of running Phase 6 evaluation step by step
    """
    
    print("\n" + "=" * 80)
    print("PHASE 6: STEP-BY-STEP EVALUATION EXAMPLE")
    print("=" * 80)
    
    # This example shows how to run individual components of Phase 6
    config_path = "configs/phase6_evaluation_config.yaml"
    output_dir = "results/phase6_step_by_step"
    checkpoint_path = "experiments/diabetic_retinopathy_phase5_20250805_194030/checkpoints/best_model.pth"
    
    pipeline = Phase6EvaluationPipeline(config_path, output_dir)
    
    print(f"\n📋 Step-by-step evaluation of: {Path(checkpoint_path).name}")
    
    # Step 1: Prepare data loaders
    print(f"\n1. Preparing data loaders...")
    try:
        data_loaders = pipeline._prepare_data_loaders()
        internal_loaders, external_loaders = pipeline._separate_internal_external_datasets(data_loaders)
        print(f"   ✓ Internal datasets: {list(internal_loaders.keys())}")
        print(f"   ✓ External datasets: {list(external_loaders.keys())}")
    except Exception as e:
        print(f"   ✗ Failed to prepare data loaders: {e}")
        return
    
    # Step 2: Comprehensive performance evaluation (Step 6.1)
    print(f"\n2. Step 6.1: Comprehensive Performance Evaluation...")
    try:
        internal_results = pipeline.comprehensive_evaluator.evaluate_model_checkpoint(
            checkpoint_path, internal_loaders, pipeline.config.get('model')
        )
        print(f"   ✓ Internal evaluation completed for {len(internal_results)} datasets")
    except Exception as e:
        print(f"   ✗ Step 6.1 failed: {e}")
        return
    
    # Step 3: External validation (Step 6.2)
    print(f"\n3. Step 6.2: External Validation & Generalization Testing...")
    try:
        external_validation_results = pipeline.external_validator.run_external_validation(
            checkpoint_path, internal_results, external_loaders, pipeline.config.get('model')
        )
        print(f"   ✓ External validation completed")
        
        if 'generalization_analysis' in external_validation_results:
            gen_analysis = external_validation_results['generalization_analysis']
            if 'overall_analysis' in gen_analysis:
                overall = gen_analysis['overall_analysis']
                print(f"   ✓ Generalization risk: {overall['risk_assessment']['overall_risk']}")
    except Exception as e:
        print(f"   ✗ Step 6.2 failed: {e}")
        return
    
    # Step 4: Explainability analysis (Step 6.3)
    print(f"\n4. Step 6.3: Explainability & Visualization...")
    try:
        explainability_results = pipeline.explainability_analyzer.run_comprehensive_explainability_analysis(
            checkpoint_path, data_loaders, pipeline.config.get('model'), n_samples=500
        )
        print(f"   ✓ Explainability analysis completed")
        
        if 'insights' in explainability_results:
            insights = explainability_results['insights']
            print(f"   ✓ Generated insights for model interpretability")
    except Exception as e:
        print(f"   ✗ Step 6.3 failed: {e}")
        return
    
    # Step 5: Generate visualizations
    print(f"\n5. Generating comprehensive visualizations...")
    try:
        all_evaluation_results = {**internal_results, **external_validation_results['external_results']}
        
        visualization_results = pipeline.visualization_generator.generate_comprehensive_visualization_suite(
            all_evaluation_results,
            training_history=pipeline._load_training_history(checkpoint_path),
            comparison_results=None
        )
        print(f"   ✓ Visualization suite generated")
    except Exception as e:
        print(f"   ✗ Visualization generation failed: {e}")
        return
    
    print(f"\n✅ Step-by-step evaluation completed successfully!")
    print(f"Results saved to: {output_dir}")


def main():
    """Main function for running examples"""
    
    print("Phase 6 Evaluation Examples")
    print("Choose an example to run:")
    print("1. Complete Phase 6 evaluation pipeline")
    print("2. Step-by-step evaluation")
    print("3. Both examples")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        example_phase6_evaluation()
    elif choice == "2":
        example_step_by_step_evaluation()
    elif choice == "3":
        example_phase6_evaluation()
        example_step_by_step_evaluation()
    else:
        print("Invalid choice. Running complete evaluation by default.")
        example_phase6_evaluation()


if __name__ == "__main__":
    main()
