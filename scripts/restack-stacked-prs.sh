#!/usr/bin/env bash
# Cascades a branch's current tip through every open PR stacked on top of it, in order.
#
# A base branch picking up new commits (a fix, a rebase onto dev, anything) never retriggers CI for a PR whose base is that branch: GitHub's pull_request `synchronize` event only fires when a PR's own head ref changes, not when its base does. Left alone, every PR stacked on top of a fixed branch keeps showing its old, stale, possibly-failing CI run indefinitely. This script does the merge-forward-and-push that would otherwise have to happen by hand, one branch at a time, top to bottom of the stack.
#
# Usage: scripts/restack-stacked-prs.sh <bottom-branch>
#
# Walks the chain by asking GitHub which open PR (if any) is based on each branch in turn (gh pr list --base), rather than a hardcoded branch list, so it stays correct as PRs are added to or merged out of the stack. Stops at the first real merge conflict instead of skipping past it: every branch above a broken link depends on content that link never actually delivered, so cascading past it would push half-updated branches.
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <bottom-branch>" >&2
  exit 1
}

[[ $# -eq 1 ]] || usage
root_branch="$1"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "restack-stacked-prs: working tree is not clean, refusing to check out branches over it" >&2
  git status --short >&2
  exit 1
fi

echo "== walking the stack rooted at ${root_branch} =="
chain=("$root_branch")
current="$root_branch"
while true; do
  next=$(gh pr list --base "$current" --state open --json headRefName --jq '.[0].headRefName // empty')
  [[ -z "$next" ]] && break
  chain+=("$next")
  current="$next"
done

if [[ "${#chain[@]}" -eq 1 ]]; then
  echo "restack-stacked-prs: no open PR is based on ${chain[0]}, nothing to cascade"
  exit 0
fi

echo "== chain: ${chain[*]} =="
git fetch origin "${chain[@]}" --quiet

for ((i = 1; i < ${#chain[@]}; i++)); do
  prev="${chain[$((i - 1))]}"
  cur="${chain[$i]}"
  echo "== merging ${prev} into ${cur} =="
  git checkout --quiet -B "$cur" "origin/$cur"
  if git merge --no-edit "origin/$prev"; then
    git push origin "$cur"
    echo "${cur}: merged and pushed"
  else
    conflicted=$(git diff --name-only --diff-filter=U)
    echo "restack-stacked-prs: ${cur} conflicts with ${prev}, stopping the cascade here. Conflicting paths:" >&2
    echo "${conflicted//$'\n'/$'\n'  }" >&2
    git merge --abort
    echo "restack-stacked-prs: resolve ${cur} by hand, push it, then re-run against ${cur} to continue the rest of the stack" >&2
    exit 1
  fi
done
