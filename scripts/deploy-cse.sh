#!/usr/bin/env bash
# Builds the site and copies it to https://servingstudio.cs.washington.edu/,
# whose files live in the CSE web area that any CSE web host (bicycle,
# recycle) mounts.
#
#   scripts/deploy-cse.sh            build, then sync
#   scripts/deploy-cse.sh --dry-run  build, then list what the sync would change
#
# Needs passwordless ssh to $CSE_HOST.
set -euo pipefail

CSE_HOST="${CSE_HOST:-bicycle}"
CSE_DIR="${CSE_DIR:-/cse/web/research/servingstudio}"

cd "$(dirname "$0")/.."

npm run build

# --delete keeps the server an exact copy of the build. Owner, group and mode
# are not copied: the directory is setgid tech_cs, and Apache needs files
# world-readable whatever the local umask was.
rsync -rlt --delete --itemize-changes \
  --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  "$@" dist/ "$CSE_HOST:$CSE_DIR/"
