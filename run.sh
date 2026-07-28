#!/usr/bin/env bash
# One-command driver for the vLLM Gemma eager-import repro.
#
# It builds a single editable vLLM install (Python-only, via
# VLLM_USE_PRECOMPILED so no CUDA compile is needed), then swaps between the
# stock main commit and the fixed branch with `git checkout` and runs repro.py
# against each -- once with a REAL old transformers (5.4.0, which genuinely
# lacks transformers.models.gemma4) and once with a modern transformers plus
# the deterministic --simulate monkeypatch.
#
# Usage:  ./run.sh
# Output: logs under ./evidence/
set -u

# --- pinned refs (override via env) ------------------------------------------
STOCK_SHA="${STOCK_SHA:-60b3d39cd36c53a698040edbf51406d3febc97a7}"   # upstream main
FIXED_SHA="${FIXED_SHA:-b57f01be01c8bca2339d1b17be4e280ec5009bfd}"   # rebased fix
FIXED_REMOTE="${FIXED_REMOTE:-https://github.com/jethac/vllm.git}"
OLD_TF="${OLD_TF:-5.4.0}"        # real transformers lacking gemma4 (5.5.0 first shipped it)
NEW_TF="${NEW_TF:-5.10.2}"       # modern transformers that ships gemma4
VLLM_DIR="${VLLM_DIR:-$PWD/_vllm_src}"

HERE="$(cd "$(dirname "$0")" && pwd)"
EVID="$HERE/evidence"
mkdir -p "$EVID"
# Clear any previously-committed arm logs so a re-run cannot be confused with
# stale evidence (run_driver.log is regenerated out of band and left alone).
rm -f "$EVID"/stock_*.log "$EVID"/fixed_*.log

log() { echo "[run.sh] $*"; }

# --- clone vLLM & fetch both refs --------------------------------------------
if [ ! -d "$VLLM_DIR/.git" ]; then
  log "cloning vLLM main -> $VLLM_DIR"
  git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$VLLM_DIR"
fi
cd "$VLLM_DIR"
log "fetching stock $STOCK_SHA and fixed $FIXED_SHA"
git fetch --depth 200 origin "$STOCK_SHA"
git fetch "$FIXED_REMOTE" "$FIXED_SHA"

# --- editable, precompiled (Python-only) install -----------------------------
git checkout -q "$STOCK_SHA"
export VLLM_USE_PRECOMPILED=1
log "installing vLLM (editable, precompiled). This downloads a wheel; CPU is fine."
pip install --upgrade pip setuptools wheel >/dev/null
pip install -e . 2>&1 | tail -5
pip install "transformers==$OLD_TF" 2>&1 | tail -3

run_arm() {  # <sha> <label> <extra repro args...>
  local sha="$1"; shift
  local label="$1"; shift
  cd "$VLLM_DIR"; git checkout -q "$sha"
  local out="$EVID/${label}.log"
  log "ARM: $label (checkout $sha) -> evidence/${label}.log"
  {
    echo "############################################################"
    echo "# repro arm: $label"
    echo "# vllm ref : $sha"
    echo "# cmd      : python repro.py $*"
    echo "# date     : $(date -u +%FT%TZ)"
    echo "############################################################"
    ( cd "$HERE" && python repro.py "$@" )
    echo "[exit code] $?"
  } 2>&1 | tee "$out"
}

# --- REAL old-transformers arm (headline): only gemma4 is genuinely absent in
#     transformers 5.4.0 (gemma3 / gemma3n shipped earlier). --------------------
python -c "import transformers;print('[run.sh] transformers',transformers.__version__)" 2>&1 | tail -1
run_arm "$STOCK_SHA" "stock_real_gemma4_tf$OLD_TF" --arch gemma4
run_arm "$FIXED_SHA" "fixed_real_gemma4_tf$OLD_TF" --arch gemma4

# --- deterministic monkeypatch arms on a modern transformers, covering all
#     three modules this fix touches. -------------------------------------------
pip install "transformers==$NEW_TF" 2>&1 | tail -3
python -c "import transformers;print('[run.sh] transformers',transformers.__version__)" 2>&1 | tail -1
for arch in gemma4 gemma3n gemma3; do
  run_arm "$STOCK_SHA" "stock_sim_${arch}_tf$NEW_TF" --arch "$arch" --simulate
  run_arm "$FIXED_SHA" "fixed_sim_${arch}_tf$NEW_TF" --arch "$arch" --simulate
done

log "done. evidence in $EVID/"
ls -la "$EVID"
