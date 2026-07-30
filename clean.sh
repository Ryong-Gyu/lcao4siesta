#!/usr/bin/env bash

# Remove reproducible Python build, test, and bytecode artifacts before staging.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

dry_run=false
case "${1:-}" in
    "")
        ;;
    -n|--dry-run)
        dry_run=true
        ;;
    -h|--help)
        printf 'Usage: %s [--dry-run]\n' "${0##*/}"
        exit 0
        ;;
    *)
        printf 'Unknown option: %s\n' "$1" >&2
        printf 'Usage: %s [--dry-run]\n' "${0##*/}" >&2
        exit 2
        ;;
esac

artifacts=()
while IFS= read -r -d '' artifact; do
    artifacts+=("$artifact")
done < <(
    find . \
        -path './.git' -prune -o \
        -type d \( \
            -name '__pycache__' -o \
            -name '.pytest_cache' -o \
            -name '.mypy_cache' -o \
            -name '.ruff_cache' -o \
            -name '.tox' -o \
            -name '.nox' -o \
            -name '*.egg-info' -o \
            -name '.ipynb_checkpoints' \
        \) -prune -print0 -o \
        -type d \( -path './build' -o -path './dist' -o -path './htmlcov' \) \
            -prune -print0 -o \
        -type f \( \
            -name '*.py[co]' -o \
            -name '*.nbc' -o \
            -name '*.nbi' -o \
            -path './.coverage' -o \
            -path './.coverage.*' \
        \) -print0
)

if ((${#artifacts[@]} == 0)); then
    printf 'No generated artifacts found.\n'
    exit 0
fi

if "$dry_run"; then
    printf 'Would remove:\n'
else
    printf 'Removing:\n'
fi

for artifact in "${artifacts[@]}"; do
    printf '  %s\n' "$artifact"
    if ! "$dry_run"; then
        rm -rf -- "$artifact"
    fi
done

printf '%s %d artifact(s).\n' \
    "$("$dry_run" && printf 'Found' || printf 'Removed')" \
    "${#artifacts[@]}"
