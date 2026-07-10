#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apply_patch_once() {
  local repo="$1"
  local patch="$2"
  if git -C "${repo}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
    echo "already applied: ${patch}"
  elif git -C "${repo}" apply --check "${patch}"; then
    git -C "${repo}" apply "${patch}"
    echo "applied: ${patch}"
  else
    echo "cannot apply ${patch}; submodule is not at the pinned clean revision" >&2
    exit 1
  fi
}

git -C "${ROOT}" submodule update --init --recursive
apply_patch_once "${ROOT}/third_party/openpi" "${ROOT}/patches/openpi.patch"
apply_patch_once "${ROOT}/third_party/RLinf" "${ROOT}/patches/rlinf.patch"
cp -a "${ROOT}/overlays/openpi/." "${ROOT}/third_party/openpi/"
cp -a "${ROOT}/overlays/rlinf/." "${ROOT}/third_party/RLinf/"

echo "PostVLA overlays are installed."

