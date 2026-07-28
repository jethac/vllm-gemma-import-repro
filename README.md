# vLLM Gemma multimodal modules crash on import with an older transformers

Minimal, runnable reproduction for a vLLM bug: the **Gemma 3 / 3n / 4 multimodal
model modules eagerly import version-gated `transformers.models.<gemma*>` symbols
at module top level**, so on a `transformers` too old to ship those submodules
the import raises `ModuleNotFoundError` **at import time** — which breaks vLLM's
architecture inspection for the whole Gemma family, not just when a Gemma model
is actually loaded.

- **vLLM PR:** _<!-- TODO: fill in PR # once filed -->_
- **Fixed branch:** [`jethac/vllm@fix/gemma-mm-lazy-transformers-imports`](https://github.com/jethac/vllm/tree/fix/gemma-mm-lazy-transformers-imports)
  (rebased on upstream `vllm-project/vllm` main).
- The live evidence logs referenced below are in [`evidence/`](evidence/) — real
  stdout captured on a clean cloud box, not hand-written.

## The bug

Stock vLLM main (`vllm/model_executor/models/gemma4_mm.py`) does, at the top of
the module:

```python
from transformers.models.gemma4 import (
    Gemma4Config, Gemma4Processor, Gemma4VisionConfig)
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4AudioConfig, Gemma4TextConfig)
```

The same pattern exists in three more modules:

| module | eager top-level import (stock) |
|---|---|
| `gemma4_mm.py:26,31` | `transformers.models.gemma4` (+ `.configuration_gemma4`) |
| `gemma3n_mm.py:9,17` | `transformers.models.gemma3n`, `transformers.models.siglip` |
| `gemma3n.py:22`      | `transformers.models.gemma3n.configuration_gemma3n` |
| `gemma3_mm.py:10,11` | `transformers.models.gemma3.image_processing_gemma3` (torchvision-backed), `.processing_gemma3` |

### Why it's more than "you need a newer transformers"

vLLM's model **registry inspects an architecture by importing its model
module** (`ModelRegistry` → `load_model_cls` → `importlib.import_module(...)`).
So the eager import fires during *inspection*, before any model is loaded, and:

- it breaks inspection of the arch itself (`Gemma4ForConditionalGeneration`), and
- it breaks **every module that imports the failing one** —
  `gemma4_unified` and (in the DiffusionGemma fork) `diffusion_gemma` both
  `import ... gemma4_mm`; `gemma3n_mm` imports the `gemma3n` text backbone.

This is the generalized form of what blocked DiffusionGemma serving on a box
whose `transformers` predated Gemma 4.

### The version boundary (real, not synthetic)

`transformers.models.gemma4` **first shipped in `transformers==5.5.0`**
(confirmed by wheel inspection: `5.4.0` lacks `models/gemma4/`, `5.5.0` has it).
vLLM main only pins `transformers >= 5.5.3`, so an install that is barely behind
— e.g. `transformers==5.4.0` — genuinely lacks the submodule and triggers the
crash. The repro uses **`transformers==5.4.0` as the real old version**, and also
supports a deterministic `--simulate` monkeypatch (a `None` entry in
`sys.modules`) that works on any transformers.

## The fix

Move the version-gated imports out of module top level: keep them under
`TYPE_CHECKING` for annotations and import them inside the
`get_hf_config` / `get_hf_processor` / `get_num_crops` helpers (and the gemma3n
`isinstance` check). The modules then import and register on older transformers;
the missing dependency surfaces as a **clear `ModuleNotFoundError` naming the
submodule, only when the processor is actually invoked**. No GPU / accuracy /
serving-behavior change — serving Gemma 4 still requires a new-enough
transformers. 5 files, +219/−36, plus a regression test.

## Reproduce

### One command (clean Linux box, CPU is fine)

```bash
git clone https://github.com/jethac/vllm-gemma-import-repro.git
cd vllm-gemma-import-repro
./run.sh          # builds one editable, precompiled (no-CUDA-compile) vLLM,
                  # swaps stock<->fixed via git checkout, writes evidence/*.log
```

`run.sh` installs vLLM once with `VLLM_USE_PRECOMPILED=1 pip install -e .`
(Python-only, downloads a prebuilt wheel — no CUDA toolchain, no GPU needed for
this import/registry bug), then runs four arms:

| log | vLLM | transformers | expect |
|---|---|---|---|
| `evidence/stock_real_transformers-5.4.0.log` | stock main | 5.4.0 (real, no gemma4) | **crash** (exit 2) |
| `evidence/fixed_real_transformers-5.4.0.log` | fixed | 5.4.0 (real, no gemma4) | clean import + register; deferred error (exit 0) |
| `evidence/stock_simulated_transformers-5.10.2.log` | stock main | 5.10.2 + `--simulate` | **crash** (exit 2) |
| `evidence/fixed_simulated_transformers-5.10.2.log` | fixed | 5.10.2 + `--simulate` | clean (exit 0) |

### Just the check (against a vLLM already on your path)

```bash
python repro.py                 # observes whatever vLLM is importable
python repro.py --simulate      # forces the gemma4 submodule "absent"
python repro.py --arch Gemma3nForConditionalGeneration --simulate
```

`repro.py` reports what actually happens and exits:
`2` = stock eager-import crash reproduced, `0` = fixed (imports + registers,
error deferred to processor use), `3` = unexpected.

## Expected output

**Stock + missing gemma4** (`ARM A`) — architecture inspection crashes at import:

```
ARM A -- import vllm.model_executor.models.gemma4_mm
RESULT: ModuleNotFoundError: No module named 'transformers.models.gemma4'
  --> STOCK BUG: eager top-level import failed at import time.
ARM B -- transitive victim: import vllm.model_executor.models.gemma4_unified
RESULT: ModuleNotFoundError: No module named 'transformers.models.gemma4'
  --> vllm...gemma4_unified is broken too, because it imports ...gemma4_mm.
VERDICT
STOCK vLLM: version-gated eager import crashes at import time, ...
[exit code] 2
```

**Fixed + missing gemma4** — imports, registers, defers the error:

```
ARM A -- import vllm.model_executor.models.gemma4_mm
RESULT: imported cleanly (vllm.model_executor.models.gemma4_mm).
ARM B -- architecture still registers
'Gemma4ForConditionalGeneration' in ModelRegistry.get_supported_archs(): True
ARM C -- missing dependency surfaces only when the processor is used
RESULT: ImportError: No module named 'transformers.models.gemma4'
  --> clear, deferred error naming the missing submodule, raised only on processor use.
VERDICT
FIXED vLLM: module imports and registers even without the version-gated
transformers submodule; ...
[exit code] 0
```

See [`evidence/`](evidence/) for the verbatim captured runs.
