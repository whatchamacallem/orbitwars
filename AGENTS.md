# Kaggle

Do not push a notebook to kaggle unless explicitly asked.

Assume all edits are to "orbitwars.ipynb" unless otherwise specified.

All code and text has to be formatted so it is human readable. Code stored in
strings has to be broken out across multiple lines.

Credentials: `KAGGLE_API_TOKEN` env var is already in `~/.kaggle/kaggle.json` as
`{"username":"ajohnston7354","key":"..."}`.

Push: `kaggle kernels push -p .` from repo root (requires `kernel-metadata.json`
— update `id`, `title`, and `code_file` to match the target notebook before
pushing).
