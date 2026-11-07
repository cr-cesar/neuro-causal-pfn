#!/bin/bash
# Brief 9: pack the aggregate outputs of every task for the return trip.
# Everything under R (the report folder the runbook writes to) except model,
# pool, image and log files; refuses if any packed text file names an image
# (".nii") or holds a column called filename / file / files.
#
#   R=$HOME/p2_bcb/brief9_report bash scripts/audit_20261107/collect_results.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
R="${R:?set R to the report folder}"
OUT="${OUT:-$HOME/brief9_results_$(date +%Y%m%d_%H%M).tar.gz}"
cd "$R"
mapfile -t files < <(find . -type f \( -name "*.csv" -o -name "*.txt" -o -name "*.md" \) | LC_ALL=C sort)
[ "${#files[@]}" -gt 0 ] || { echo "nothing to pack in $R" >&2; exit 1; }
bad=0
for f in "${files[@]}"; do
    if grep -qiE "\.nii" "$f"; then echo "refusing: $f mentions an image file" >&2; bad=1; fi
    if head -1 "$f" | tr ',' '\n' | grep -qixE "filename|file|files|subject|patient|patient_id"; then
        echo "refusing: $f has a per-file / per-patient column" >&2; bad=1
    fi
done
[ "$bad" = 0 ] || exit 1
{
    echo "packed $(date -Iseconds) on $(hostname -s)"
    echo "code $(git -C "$HERE" rev-parse --short HEAD) ($(git -C "$HERE" rev-parse --abbrev-ref HEAD)), local changes: $(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)"
    for f in "${files[@]}"; do echo "$(md5sum "$f" | cut -c1-12)  $f"; done
} > MANIFEST.txt
tar -czf "$OUT" MANIFEST.txt "${files[@]}"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1), ${#files[@]} files)"
