# Diabetic Retinopathy

## Installation

1. **Install uv**:

   ```bash
   # Using pip
   pip install uv

   # Or using curl (Linux/macOS)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or using PowerShell (Windows)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
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

```bash
uv run python run_pipeline5_training_optimization.py --mode quick --experiment-name test_real_training
```

**Pipeline 6 - Evaluation & Analysis**:

```bash
uv run python run_pipeline6_evaluation_analysis.py
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
