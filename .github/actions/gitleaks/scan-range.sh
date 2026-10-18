#!/usr/bin/env bash
# Prints the git log range gitleaks scans: the commits a push or a pull request adds, and nothing the base branch already had.
#
# Inputs, all from the environment: EVENT_NAME (github.event_name), GITHUB_SHA, PUSHED_FROM (github.event.before). Run it inside the repository; a subdirectory of it is fine.
#
# Every event needs the full history (actions/checkout with fetch-depth: 0), so a shallow checkout is refused. git log stops at the shallow boundary without an error and shows the boundary commit as adding its whole tree, so any range scans the wrong changes: at depth 2 a three-commit pull request's merge commit still has both parents, but only its newest commit is in the range, and that commit appears to add every file in the repository, base branch included.
#
# On a pull_request event GITHUB_SHA is the merge commit GitHub built for the run (refs/pull/N/merge), and its first parent is the base branch tip the pull request is merged onto, so the range is exactly the pull request's own commits. github.event.pull_request.base.sha is not used: GitHub does not move it when the base branch advances, so a range from it also takes in every commit merged to the base branch since, and a finding in one of those fails every pull request opened before it. A re-run reuses the run's merge commit, whose first parent is the base tip it was built on, and a reopened or updated pull request gets a newly built merge commit on the current tip. A GITHUB_SHA with fewer than two parents is refused rather than scanned as a single commit, since it is not a merge commit.
#
# On a push the range is the commits since the previous tip (github.event.before). Two pushes have no previous tip in the checkout, and both scan the tip commit alone: a push that created the branch (before is all zeros), and a force-push whose previous tip is no longer reachable from any ref (git log would reject the range, and gitleaks reports "0 commits scanned" and passes, so a commit the force-push added would go unscanned). The tip is the only commit such a push is certain to have added. The workflows calling this run on push to the default branch only, so there is no other branch whose commits could be subtracted instead, and a scan of the whole history would fail on any finding already on that branch. A branch's other commits are covered by its pull request's scan; the force-push case also prints a warning naming the previous tip, so the narrower scan shows in the run. Any other event is refused, since it has no "new commits" to scan.
set -euo pipefail

if [[ "$(git rev-parse --is-shallow-repository)" == true ]]; then
  echo "::error::this checkout is shallow, so a commit range would silently stop at its boundary; check out the full history (fetch-depth: 0)" >&2
  exit 1
fi

case "$EVENT_NAME" in
  pull_request)
    if ! base="$(git rev-parse --verify --quiet "${GITHUB_SHA}^1")" || ! git rev-parse --verify --quiet "${GITHUB_SHA}^2" > /dev/null; then
      echo "::error::$GITHUB_SHA has fewer than two parents, so it is not a pull request merge commit" >&2
      exit 1
    fi
    echo "${base}..${GITHUB_SHA}"
    ;;
  push)
    if [[ "$PUSHED_FROM" == 0000000000000000000000000000000000000000 ]]; then
      echo "-1 ${GITHUB_SHA}"
    elif ! git cat-file -e "${PUSHED_FROM}^{commit}" 2> /dev/null; then
      echo "::warning::the previous tip $PUSHED_FROM is not in this checkout (a force-push), so only the tip commit $GITHUB_SHA is scanned" >&2
      echo "-1 ${GITHUB_SHA}"
    else
      echo "${PUSHED_FROM}..${GITHUB_SHA}"
    fi
    ;;
  *)
    echo "::error::no commit range is defined for the $EVENT_NAME event; the gitleaks scan runs on pull_request and push" >&2
    exit 1
    ;;
esac
