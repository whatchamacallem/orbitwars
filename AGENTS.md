# Kaggle

Assume all edits are to "orbitwars.ipynb" unless otherwise specified.

read rules.md for all the rules.

Do not push a notebook to kaggle unless explicitly asked.

All code and text has to be formatted so it is human readable. Code stored in
strings has to be broken out across multiple lines. In notebooks, cell `source`
must be a JSON array of strings (one per line, each ending with `\n` except the
last), never a single concatenated string.

Use object as keys and not object ids.

Never delete or modify comments or code except to update or modify as requested.
Do not delete dead code. Do not delete Python documentation """ like this """.

Do not put underscores on the beginning of function names.

The `NotebookEdit` tool collapses `source` into a single string. After every
`NotebookEdit` call, fix the affected cell by splitting the string on newlines
and converting it back to an array (each line ending with `\n` except the last),
then write it back with `json.dump`.

No unicode characters anywhere in notebooks or source files. Use plain ASCII
only (e.g. `-` not `─`, `->` not `→`). In strings use "'" not "\"".

Credentials: `KAGGLE_API_TOKEN` env var is already in `~/.kaggle/kaggle.json` as
`{"username":"ajohnston7354","key":"..."}`.

Push: `kaggle kernels push -p .` from repo root (requires `kernel-metadata.json`
-- update `id`, `title`, and `code_file` to match the target notebook before
pushing).
