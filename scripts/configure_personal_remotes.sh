#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <github-user> [https|ssh]" >&2
  exit 1
fi

github_user="$1"
protocol="${2:-https}"
workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$protocol" in
  https|ssh)
    ;;
  *)
    echo "Protocol must be 'https' or 'ssh'." >&2
    exit 1
    ;;
esac

repo_names=("DemoGen" "robomimic" "robosuite")
upstream_urls=(
  "https://github.com/TEA-Lab/DemoGen.git"
  "https://github.com/ARISE-Initiative/robomimic.git"
  "https://github.com/ARISE-Initiative/robosuite.git"
)

make_origin_url() {
  local repo_name="$1"
  if [[ "$protocol" == "ssh" ]]; then
    printf 'git@github.com:%s/%s.git\n' "$github_user" "$repo_name"
  else
    printf 'https://github.com/%s/%s.git\n' "$github_user" "$repo_name"
  fi
}

ensure_remote() {
  local repo_path="$1"
  local remote_name="$2"
  local remote_url="$3"

  if git -C "$repo_path" remote get-url "$remote_name" >/dev/null 2>&1; then
    git -C "$repo_path" remote set-url "$remote_name" "$remote_url"
  else
    git -C "$repo_path" remote add "$remote_name" "$remote_url"
  fi
}

for i in "${!repo_names[@]}"; do
  repo_name="${repo_names[$i]}"
  repo_path="$workspace_root/repos/$repo_name"
  upstream_url="${upstream_urls[$i]}"
  origin_url="$(make_origin_url "$repo_name")"

  if [[ ! -d "$repo_path/.git" ]]; then
    echo "Skip $repo_name: not a git repo at $repo_path" >&2
    continue
  fi

  current_origin="$(git -C "$repo_path" remote get-url origin 2>/dev/null || true)"
  has_upstream=0
  if git -C "$repo_path" remote get-url upstream >/dev/null 2>&1; then
    has_upstream=1
  fi

  if [[ "$current_origin" == "$upstream_url" && "$has_upstream" -eq 0 ]]; then
    git -C "$repo_path" remote rename origin upstream
  fi

  ensure_remote "$repo_path" upstream "$upstream_url"
  ensure_remote "$repo_path" origin "$origin_url"

  echo "$repo_name"
  echo "  upstream -> $(git -C "$repo_path" remote get-url upstream)"
  echo "  origin   -> $(git -C "$repo_path" remote get-url origin)"
done
