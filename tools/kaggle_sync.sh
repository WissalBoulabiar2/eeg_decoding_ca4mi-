#!/bin/bash
# Keep CA4MI-MP results outside Kaggle: small .txt result files are pushed to the `results` branch
# of the GitHub repo after every run, and restored at the start of a new session.
#   bash tools/kaggle_sync.sh restore <results_dir>
#   bash tools/kaggle_sync.sh push    <results_dir>
# Needs GH_TOKEN in the environment (GitHub token with write access to this repo, from Kaggle Secrets).
# The token is never printed.
set -e
MODE="$1"; DIR="$2"
[ -n "$GH_TOKEN" ] || { echo "kaggle_sync: GH_TOKEN not set, nothing synced"; exit 0; }
URL="https://x-access-token:${GH_TOKEN}@github.com/WissalBoulabiar2/eeg_decoding_ca4mi-.git"
R=/kaggle/working/.results_repo
NAME=$(basename "$DIR")

if [ ! -d "$R/.git" ]; then
  if ! git clone -q --branch results "$URL" "$R" 2>/dev/null; then
    rm -rf "$R"; mkdir -p "$R"; cd "$R"; git init -q; git checkout -q -b results; git remote add origin "$URL"
  fi
fi

if [ "$MODE" = "restore" ]; then
  if [ -d "$R/$NAME" ]; then
    mkdir -p "$DIR"; cp -rn "$R/$NAME/." "$DIR/"
    echo "kaggle_sync: restored $(find "$DIR" -name 'val_accuracy_matrix.txt' | wc -l) finished runs into $DIR"
  else
    echo "kaggle_sync: no previous results for $NAME"
  fi
elif [ "$MODE" = "push" ]; then
  cd "$(dirname "$DIR")"
  find "$NAME" -name '*.txt' -exec cp --parents {} "$R/" \;
  cd "$R"; git add -A
  git -c user.name=kaggle -c user.email=kaggle@users.noreply.github.com commit -qm "results $(date -u +%F_%H%M)" || true
  git push -q origin results && echo "kaggle_sync: pushed $(find "$NAME" -name 'val_accuracy_matrix.txt' | wc -l) finished runs to GitHub (branch results)"
fi
