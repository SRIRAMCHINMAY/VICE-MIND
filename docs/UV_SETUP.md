# Development With uv

`uv` now manages the Python project environment and dependency lockfile. The
project metadata lives in `pyproject.toml`; exact resolved versions live in
`uv.lock`.

## Install uv

Use the official installer or your operating system's package manager. Verify:

```bash
uv --version
```

## Create the Environment

From the repository root:

```bash
uv python install 3.11
uv sync --group dev
```

`uv sync` creates `.venv`, installs the locked dependencies, and keeps the
environment reproducible. Python 3.11 is the safest default for this project;
the declared project range is Python 3.11 through 3.13.

## Run the Project

Generate a fresh deterministic dataset:

```bash
uv run python generate.py --count 5000 --seed 42 --out dataset/phase1.jsonl
```

Train the local model on a CUDA machine:

```bash
uv run python train.py --data dataset/phase1.jsonl --out models/tommy-phase1
```

Run one sandbox cycle after training:

```bash
TOMMY_MODEL_DIR=models/tommy-phase1 uv run python run_sandbox.py
```

Run the qualitative scenarios:

```bash
TOMMY_MODEL_DIR=models/tommy-phase1 uv run python test_scenarios.py
```

Check formatting/lint without changing files:

```bash
uv run ruff check .
uv run pytest
```

The unit tests do not require model weights. The scenario script is a separate
qualitative check and requires a trained model.

## GPU Note

The training script explicitly requires CUDA and FP16. Confirm the installation
before starting a long run:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If the default PyPI PyTorch wheel does not match your NVIDIA driver, follow the
official PyTorch selector to choose the appropriate CUDA wheel, then regenerate
the lockfile. Do not mix an arbitrary system PyTorch installation with the
locked environment.

## Updating Dependencies

Add or update dependencies through `uv`, not by manually editing `uv.lock`:

```bash
uv add package-name
uv add --dev package-name
uv lock --upgrade-package package-name
uv sync --group dev
```

Commit both `pyproject.toml` and `uv.lock`. Do not commit `.venv`, model
weights, or Hugging Face caches.

## Useful uv Documentation

- Project guide: https://docs.astral.sh/uv/guides/projects/
- `pyproject.toml` reference: https://docs.astral.sh/uv/concepts/projects/config/
- Dependency groups: https://docs.astral.sh/uv/concepts/projects/dependencies/
- Python management: https://docs.astral.sh/uv/concepts/python-versions/
- PyTorch installation selector: https://pytorch.org/get-started/locally/
