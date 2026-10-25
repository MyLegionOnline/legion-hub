#!/bin/sh
# Fails if the git tree contains a path that looks like a plan.
# Used by the export script (before the first push) and by the public
# repository's "no-plans" workflow.
#
# Usage: oss/check_no_plans.sh            # checks tracked files (git ls-files)
#        oss/check_no_plans.sh <dir>      # checks every file under <dir>
set -eu

PATTERN='(^|/)plans?(/|$)|plan[^/]*\.md$|PLAN[^/]*\.md$'

if [ "$#" -ge 1 ]; then
    found=$(cd "$1" && find . -path ./.git -prune -o -type f -print | sed 's|^\./||' | grep -E "$PATTERN" || true)
else
    found=$(git ls-files | grep -E "$PATTERN" || true)
fi

if [ -n "$found" ]; then
    echo "ERROR: plan files must not be in the public tree:" >&2
    echo "$found" >&2
    exit 1
fi
echo "ok: no plan paths"
