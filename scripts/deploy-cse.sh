#!/usr/bin/env bash
# Builds the site and copies it to https://servingstudio.cs.washington.edu/,
# whose files live in the CSE web area that any CSE web host (bicycle,
# recycle) mounts.
#
#   scripts/deploy-cse.sh            build, then sync
#   scripts/deploy-cse.sh --dry-run  build, then list what the sync would change
#
# Needs passwordless ssh to $CSE_HOST, and CSE_PUBLIC_API_TARGET: the
# http://HOST:PORT the site's .htaccess proxies /api/public/v1 to. That host is
# internal and this repository is public, so it comes from the environment
# (export it in your shell profile), never from a committed file.
set -euo pipefail

cd "$(dirname "$0")/.."

CSE_HOST="${CSE_HOST:-bicycle}"
CSE_DIR="${CSE_DIR:-/cse/web/research/servingstudio}"
target="${CSE_PUBLIC_API_TARGET:-}"
if [[ ! $target =~ ^http://[A-Za-z0-9.:-]+:[0-9]+$ ]]; then
  echo "CSE_PUBLIC_API_TARGET must be http://HOST:PORT (got '${target}')" >&2
  exit 1
fi

npm run build

# Written after the build, so only a deploy carries the internal host.
sed "s|@CSE_PUBLIC_API_TARGET@|$target|" scripts/cse.htaccess > dist/.htaccess

# --delete keeps the server an exact copy of the build. Owner, group and mode
# are not copied: the directory is setgid tech_cs, and Apache needs files
# world-readable whatever the local umask was.
rsync -rlt --delete --itemize-changes \
  --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  "$@" dist/ "$CSE_HOST:$CSE_DIR/"
