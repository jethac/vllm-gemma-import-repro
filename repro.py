#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the eager version-gated transformers import bug in vLLM's Gemma
multimodal model modules.

Background
----------
vLLM's Gemma 3 / 3n / 4 multimodal model modules import symbols from
version-gated ``transformers.models.<gemma*>`` submodules (and, for Gemma 3, a
torchvision-backed image processor) at *module top level*, e.g.:

    # vllm/model_executor/models/gemma4_mm.py (stock main)
    from transformers.models.gemma4 import (
        Gemma4Config, Gemma4Processor, Gemma4VisionConfig)
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4AudioConfig, Gemma4TextConfig)

vLLM's architecture inspection imports a model module for *any* registered
architecture. So on a transformers too old to ship the submodule (e.g.
``transformers.models.gemma4`` first shipped in transformers 5.5.0; vLLM main
only pins ``>=5.5.3``, so a 0.x-behind install lacks it), that eager import
raises ``ModuleNotFoundError`` *at import time* -- breaking inspection of the
architecture, rather than failing only when the model is actually used.

This script observes the behaviour of whatever vLLM checkout is on ``sys.path``.
It does not know whether it is running against stock or fixed vLLM; it reports
what actually happens and picks an exit code:

    exit 2  -> STOCK bug reproduced: the eager import raised at import time
    exit 0  -> FIXED behaviour: module imports + registers; the missing
               dependency surfaces only when the processor is invoked
    exit 3  -> UNEXPECTED (e.g. a different import broke first)

Two ways to make the submodule absent:
  * REAL: run with a real old transformers (e.g. 5.4.0) that genuinely lacks it.
  * SIMULATED: pass ``--simulate`` to monkeypatch ``sys.modules`` so the module
    appears absent regardless of the installed transformers (deterministic).

Only the four modules this fix touches are exercised: gemma4_mm, gemma3n_mm and
gemma3_mm (plus the gemma3n text backbone that gemma3n_mm pulls in). Modules
with their *own* separate version-gated imports (e.g. gemma4_unified, tracked by
transformers-import fix #48820) are out of scope here.
"""

import argparse
import importlib
import platform
import sys

# Per-architecture config. ``import_missing`` is the version-gated submodule the
# model module eagerly imports at top level (its absence crashes import on stock
# vLLM). ``deferred_missing`` is the submodule the processor helper imports
# lazily -- on fixed vLLM it is only hit when that helper is actually invoked.
_CASES = {
    "gemma4": {
        "module": "vllm.model_executor.models.gemma4_mm",
        "arch": "Gemma4ForConditionalGeneration",
        "import_missing": "transformers.models.gemma4",
        "deferred_missing": "transformers.models.gemma4",
        "invoke": lambda m: m.Gemma4ProcessingInfo(ctx=None).get_hf_config(),
    },
    "gemma3n": {
        "module": "vllm.model_executor.models.gemma3n_mm",
        "arch": "Gemma3nForConditionalGeneration",
        "import_missing": "transformers.models.gemma3n",
        "deferred_missing": "transformers.models.gemma3n",
        "invoke": lambda m: m.Gemma3nProcessingInfo(ctx=None).get_hf_config(),
    },
    "gemma3": {
        "module": "vllm.model_executor.models.gemma3_mm",
        "arch": "Gemma3ForConditionalGeneration",
        "import_missing": "transformers.models.gemma3.image_processing_gemma3",
        "deferred_missing": "transformers.models.gemma3.processing_gemma3",
        "invoke": lambda m: m.Gemma3ProcessingInfo(ctx=None).get_num_crops(
            image_width=1, image_height=1, processor=None, mm_kwargs={}),
    },
}


def _rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _simulate_missing(name):
    """Make ``import name`` raise, mirroring an older transformers install.

    A ``None`` entry in ``sys.modules`` makes ``import name`` raise ImportError.
    """
    for cached in list(sys.modules):
        if cached == name or cached.startswith(name + "."):
            del sys.modules[cached]
    sys.modules[name] = None


def _transformers_version():
    try:
        import transformers

        return transformers.__version__
    except Exception as e:  # pragma: no cover
        return f"<not importable: {e!r}>"


def _importable(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="gemma4", choices=list(_CASES),
                    help="which Gemma multimodal module to exercise")
    ap.add_argument("--simulate", action="store_true",
                    help="monkeypatch the submodule(s) absent (deterministic)")
    args = ap.parse_args()

    case = _CASES[args.arch]
    module_name = case["module"]
    import_missing = case["import_missing"]
    deferred_missing = case["deferred_missing"]

    _rule("ENVIRONMENT")
    print(f"python                 : {platform.python_version()}")
    print(f"transformers           : {_transformers_version()}")
    print(f"arch under test        : {case['arch']}  ({module_name})")
    real_present = _importable(import_missing)
    print(f"{import_missing!r} importable (real): {real_present}")

    mode = "SIMULATED (monkeypatch)" if args.simulate else "REAL transformers"
    print(f"mode                   : {mode}")

    if args.simulate:
        # Only the *import-time* submodule is made absent here; the processor's
        # lazily-imported submodule (which for Gemma 3 is a different one) is
        # simulated separately at call time in ARM C, mirroring the real world.
        _simulate_missing(import_missing)
    absent = args.simulate or not real_present
    print(f"target submodule absent for this run: {absent}")

    if not absent:
        print("\nNOTE: the version-gated submodule is PRESENT and not simulated,"
              " so this run cannot exhibit the bug. Re-run with --simulate or an"
              " older transformers that lacks it.")

    # ---- Arm A: import the model module (exactly what the registry does) ----
    _rule(f"ARM A -- import {module_name}")
    tag = import_missing.split(".")[2]  # e.g. 'gemma4', 'gemma3', 'gemma3n'
    try:
        importlib.import_module(module_name)
        print(f"RESULT: imported cleanly ({module_name}).")
    except ImportError as e:
        # transformers' lazy loader may wrap the failure (e.g. "Could not import
        # module 'Gemma3Processor'"), so match the family tag case-insensitively.
        if tag.lower() in str(e).lower():
            print(f"RESULT: {type(e).__name__}: {e}")
            print("  --> STOCK BUG: eager top-level import failed at import time,"
                  " breaking architecture inspection of this arch.")
            _rule("VERDICT")
            print("STOCK vLLM: version-gated eager import crashes at import time.")
            sys.exit(2)
        _rule("UNEXPECTED -- a different import failed first")
        print(f"{type(e).__name__}: {e}")
        print("  (not the gemma eager import; likely an unrelated missing dep or"
              " a transformers-version incompatibility of the base package)")
        sys.exit(3)

    # Import succeeded -> fixed behaviour. Verify registration + deferred error.
    from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402

    _rule("ARM B -- architecture still registers")
    registered = case["arch"] in ModelRegistry.get_supported_archs()
    print(f"{case['arch']!r} in ModelRegistry.get_supported_archs(): {registered}")

    _rule("ARM C -- missing dependency surfaces only when the processor is used")
    if not absent:
        print("SKIPPED: submodule present and not simulated; nothing to defer.")
        ok = registered
    else:
        module = sys.modules[module_name]
        if args.simulate:
            # With a modern transformers, importing vllm may have populated the
            # real submodule, and (for Gemma 3) the processor's submodule was not
            # simulated up front -- so assert absence now, just before use.
            _simulate_missing(deferred_missing)
        # In REAL mode the submodule is genuinely absent, so the lazy import
        # fails naturally ("No module named ...") with no monkeypatch involved.
        ok = False
        try:
            case["invoke"](module)
            print("RESULT: no error raised (unexpected).")
        except ImportError as e:
            print(f"RESULT: {type(e).__name__}: {e}")
            print("  --> clear, deferred error naming the missing submodule,"
                  " raised only on processor use.")
            ok = registered
        except Exception as e:
            print(f"RESULT: {type(e).__name__}: {e}")
            print("  --> NOT the deferred ImportError; unexpected.")

    _rule("VERDICT")
    print("FIXED vLLM: module imports and registers even without the")
    print("version-gated transformers submodule; the missing dependency is a")
    print("clear, deferred error at processor-use time -- not an import crash.")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
