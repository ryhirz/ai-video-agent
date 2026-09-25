#!/usr/bin/env bash
set -u
PY=.venv/Scripts/python.exe
PASS=0; FAIL=0; FAILS=""
for f in verify_ui verify_ops verify_semantics verify_new_materials_ui verify_vertical verify_materials verify_m2 verify_asr_guard verify_speech_delete; do
  echo "===== $f ====="
  if "$PY" "_verify/$f.py" > "_verify/_out_$f.log" 2>&1; then
    PASS=$((PASS+1)); echo "  [PASS] $f"
  else
    FAIL=$((FAIL+1)); FAILS="$FAILS $f"; echo "  [FAIL] $f (详见 _verify/_out_$f.log)"
  fi
done
echo "-----------------------------------"
echo "GATES PASSED=$PASS FAILED=$FAIL"
[ "$FAIL" -eq 0 ]
