#!/usr/bin/env bash
# One-command AEGIS-SLAM demo (PRD U10): regenerate data if needed, run one
# fault-injected supervised job, print the decision trace summary.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-Dataset/synthetic}
FRAMES=${FRAMES:-90}
FAULT=${FAULT:-motion_blur}
OUT=${OUT:-runs/demo}

if [ ! -d "$DATA/rgb" ]; then
  echo "== generating synthetic dataset =="
  python -m aegis.synth --out "$DATA" --frames 200 --no_mp4
fi

echo "== running supervised pipeline with ${FAULT} injected (${FRAMES} frames) =="
python -m aegis.run --data "$DATA" --frames "$FRAMES" --mode supervised \
  --fault "$FAULT" --out "$OUT"

echo "== decision trace summary =="
python - "$OUT/trace.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))
s = t.get("summary", {})
print(f"decisions={s.get('num_decisions')} by_diagnosis={s.get('by_diagnosis')} "
      f"by_verdict={s.get('by_verdict')}")
for d in t.get("decisions", []):
    v = (d.get("outcome") or {}).get("verdict", "pending")
    print(f"  frame {d['frame']:>4}  {d['diagnosis']:<28} -> {d['action']['tool']:<22} [{v}]")
PY

echo
echo "artifacts: $OUT/result.json  $OUT/trace.json  $OUT/trajectory.png"
echo "serve the UI with:  python -m aegis.server --port 8000"
