#!/usr/bin/env bash
# vspull — one command to get a model into a running VisionServe container.
#
# A process INSIDE the container cannot see your host folders unless they are
# mounted, so `docker exec visionserve visionserve pull ./my-gd` can never find a
# host path on its own. This wrapper bridges that gap on the host side:
#
#   vspull my-gd
#     • if a local model folder exists  -> copy it into the container + install it
#                                          (validates manifest.yaml + .onnx, then
#                                           copies into the registry /root/.models)
#     • otherwise                       -> pull "my-gd" from the curated catalog
#
# Your `docker run` stays minimal — no -v needed:
#   docker run -d --gpus all -p 11435:11435 --name visionserve mtbui2010/visionserve:latest
#
# Install (once):
#   sudo install -m 0755 deploy/vspull.sh /usr/local/bin/vspull
#   # or:  alias vspull="$PWD/deploy/vspull.sh"
#
# NOTE: the cleanest setup needs none of this — bind-mount your models folder when
# starting the container (`-v ~/.visionserve_models:/root/.models`) and any local
# model folder dropped in ~/.visionserve_models shows up directly, no copy needed.
# This helper is only for the no-bind-mount case (copy a host folder into a running
# container, or fall back to the catalog).
#
# Extra arguments are passed through to `visionserve pull`, e.g.:
#   vspull my-gd --force
#
# Config via env:
#   VS_CONTAINER  (default: visionserve)              running container name
#   VS_LOCAL_DIR  (default: $HOME/.visionserve_models) where your local model folders live
set -euo pipefail

arg="${1:-}"
container="${VS_CONTAINER:-visionserve}"
local_dir="${VS_LOCAL_DIR:-$HOME/.visionserve_models}"

if [ -z "$arg" ]; then
  echo "usage: vspull <model-name|folder> [pull flags, e.g. --force]" >&2
  echo "  installs \$VS_LOCAL_DIR/<name> if it exists, else pulls <name> from the catalog" >&2
  exit 2
fi
shift

# -t only with a terminal on stdin: under CI, cron or a pipe, `docker exec -t` fails with
# "the input device is not a TTY". -i is harmless either way.
exec_flags=(-i)
if [ -t 0 ] && [ -t 1 ]; then
  exec_flags+=(-t)
fi

# Resolve the source folder: an explicit path, or a name under VS_LOCAL_DIR.
if [ -d "$arg" ]; then
  src="$arg"
elif [ -d "$local_dir/$arg" ]; then
  src="$local_dir/$arg"
else
  src=""
fi

if [ -n "$src" ]; then
  name="$(basename "$(cd "$src" && pwd)")"
  tmp="/tmp/vspull-$name-$$"
  # Remove the in-container copy however this script ends (success, failed install, Ctrl-C).
  cleanup() { docker exec "$container" rm -rf "$tmp" >/dev/null 2>&1 || true; }
  trap cleanup EXIT
  echo ">> local model folder found: $src"
  echo ">> copying into container '$container' and installing ..."
  # "$src/." copies the folder's CONTENTS (following the folder itself if it is a symlink).
  docker exec "$container" mkdir -p "$tmp"
  docker cp -L "$src/." "$container:$tmp"
  docker exec "${exec_flags[@]}" "$container" visionserve pull "$tmp" "$@"
else
  echo ">> no local folder for '$arg' (looked in $local_dir) — pulling from catalog"
  docker exec "${exec_flags[@]}" "$container" visionserve pull "$arg" "$@"
fi
