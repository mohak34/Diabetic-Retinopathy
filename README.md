# Diabetic Retinopathy

## Installation

1. **Install uv**:

   ```bash
   # Using pip
   pip install uv

   # Or using curl (Linux/macOS)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone https://github.com/mohak34/Diabetic-Retinopathy
   cd Diabetic-Retinopathy
   ```

## Usage

### Complete Workflow

**Full training** (recommended):

```bash
uv run python main.py --mode full --experiment-name production_model_v1
```

**Quick test**:

```bash
uv run python main.py --mode quick
```

### Individual Pipelines

**Pipeline 1 - Data Processing**:

```bash
uv run python run_pipeline1_data_processing.py --mode full
```

**Pipeline 2 - Data Pipeline Setup**:

```bash
uv run python run_pipeline2_data_pipeline.py --mode full
```

**Pipeline 3 - Model Architecture**:

```bash
uv run python run_pipeline3_model_architecture.py --mode full
```

**Pipeline 4 - Training Infrastructure**:

```bash
uv run python run_pipeline4_training_infrastructure.py --experiment-name test_training --mode full
```

**Pipeline 5 - Training & Optimization**:

### For Normal Training (RECOMMENDED)

```bash
# Quick test (5-10 minutes)
uv run run_focused_training.py --mode quick

# Full training (1-2 hours)
uv run run_focused_training.py --mode full
```

### For Research Experiments (CAUTION!)

```bash
# Quick test with experimental design
uv run run_pipeline5_training_optimization.py --mode quick

# Full experimental design (WARNING: Creates 292 experiments, 36+ days runtime! on a RTX 3080 and saves models with more than 300GB disk)
uv run run_pipeline5_training_optimization.py --mode full
```

**Pipeline 6 - Evaluation & Analysis**:

```bash
uv run python run_pipeline6_evaluation_analysis.py
```

### Hyperparameter Optimization (NEW)

**Quick optimization** (12 research-based configurations, ~3 hours):

```bash
uv run python run_hyperparameter_optimization.py --mode quick
```

**Full optimization** (comprehensive parameter search):

```bash
uv run python run_hyperparameter_optimization.py --mode full
```

**Analyze optimization results**:

```bash
uv run python scripts/analyze_hyperparameter_results.py experiments/hyperopt_*/optimization_results.json
```

### Selective Execution

**Run specific phase via main**:

```bash
uv run python main.py --phase pipeline5 --experiment-name custom_training
```

**Skip completed phases**:

```bash
uv run python main.py --skip-phases pipeline1 pipeline2 --experiment-name continue_training
```
