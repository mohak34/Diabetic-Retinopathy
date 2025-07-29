# Diabetic Retinopathy Detection

This project aims to detect diabetic retinopathy from retinal images.

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
