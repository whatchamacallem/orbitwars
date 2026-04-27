# Setup

## Requirements

- Python 3.12
- `kaggle-environments >= 1.28.0`
- `kaggle >= 2.1.0`

## Installation

On WSL2 the package can be installed system-wide without privileges with `--break-system-packages`:

```bash
pip install --break-system-packages "kaggle-environments>=1.28.0" "kaggle>=2.1.0"
```

To undo the system-wide install:

```bash
pip uninstall --break-system-packages kaggle-environments
```

## Local Notebook Rendering

Install jupyterlab the same way as the other packages:

```bash
pip install --break-system-packages jupyterlab
```

To render the notebook locally instead of uploading to Kaggle. The link is in the spew.

```bash
jupyter lab --no-browser &
```

## Git and Notebook Outputs

Install `nbstripout` to automatically strip notebook outputs on `git add`:

```bash
pip install --break-system-packages nbstripout
nbstripout --install
```

Outputs are preserved locally but never committed to git.
