# Setup

## Requirements

- Python 3.12
- `kaggle-environments >= 1.28.0`

## Installation

On WSL2 the package can be installed system-wide without privileges with `--break-system-packages`:

```bash
pip install --break-system-packages "kaggle-environments>=1.28.0"
```

To undo the system-wide install:

```bash
pip uninstall --break-system-packages kaggle-environments
```
