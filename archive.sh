#!/bin/sh
# SPDX-FileCopyrightText: © 2026 Adrian Johnston.
# SPDX-License-Identifier: MIT

PROJECT="$(basename $PWD)"
DATE="$(date +%Y-%m-%d)"
ARCHIVE="$PROJECT-$DATE.git.txz"

if [ ! -d ".git" ]; then
	echo "error: .git not found" >&2
	exit 1
fi

if [ "$(command ls)" = "archive.sh" ]; then
	git restore .
	echo "extracted all files in $PROJECT."

	FS_TYPE=$(stat -f -c "%T" . 2>/dev/null)
	if [ "$FS_TYPE" = "v9fs" ] || [ "$FS_TYPE" = "fuseblk" ] || [ "$FS_TYPE" = "ntfs" ]; then
		echo "Windows detected. Setting config core.fileMode false."
		git config core.fileMode false
	fi

	exit 0
fi

if [ -d "$HOME/Backups" ]; then
	DESTINATION="$HOME/Backups"
else
	DESTINATION="$HOME"
fi

tar -cJf "$DESTINATION/$ARCHIVE" -C ".." "$PROJECT/archive.sh" "$PROJECT/.git"

ls -hs1 "$DESTINATION/$ARCHIVE"
