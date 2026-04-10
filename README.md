# signalfoundry

SignalFoundry is a multi-package Python workspace for quantitative research, feature engineering, labeling, model development, and trading system execution.

## Overview

This repository is organized as a `uv` workspace containing reusable packages that can be developed and tested together.

Workspace packages:

- `core` - strategy and execution primitives
- `features` - feature engineering and indicators
- `labelling` - target/label generation
- `ml` - classical ML workflows and models
- `mt5` - MetaTrader 5 integration utilities
- `neural-net` - deep learning model components
- `pipeline` - dataset and feature pipelines
- `research` - research-facing support code
- `utils` - shared utility functions

## Prerequisites

- Python 3.12 (recommended: 3.12.9)
- `uv` installed and available on PATH

## Common Commands

From repository root:

```powershell
python build.py --sync
python build.py --test
python build.py --lint
python build.py --build
```

Full pipeline:

```powershell
python build.py --all
```

## CI

A minimal GitHub Actions workflow is included at:

- `.github/workflows/ci.yml`

It runs workspace sync and a basic import smoke test on pushes and pull requests.
